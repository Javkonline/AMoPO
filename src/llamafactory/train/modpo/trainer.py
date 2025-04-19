# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch, re
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps

import pdb

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin, PreTrainedTokenizer

    from ...hparams import FinetuningArguments


class CustomMODPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        reward_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        reward_tokenizer_modpo: Optional[Union["PreTrainedTokenizer", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self.reward_model = reward_model
        self.reward_tokenizer_modpo = reward_tokenizer_modpo
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss
    
    def modpo_loss(
        self, 
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor, 
        chosen_margin_reward: torch.FloatTensor, 
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        self.w = (self.finetuning_args.w, 1-self.finetuning_args.w)
        self.w = torch.tensor(self.w)
        chosen_rewards   = (1/self.w[0])*(self.beta * (policy_chosen_logps   - reference_chosen_logps)   - chosen_margin_reward   @ self.w[1:])
        rejected_rewards = (1/self.w[0])*(self.beta * (policy_rejected_logps - reference_rejected_logps) - rejected_margin_reward @ self.w[1:])

        logits = chosen_rewards - rejected_rewards
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()
    
    def compute_reward_model_score(
            self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        r"""
        compute the reward model score of chosen and rejected
        """
        if not self.finetuning_args.reward_model_modpo:
            return None, None
        self.reward_model.to(self.accelerator.device)
        # self.reward_model.device = self.accelerator.device
        prompt_ids = batch['prompt']
        response_ids = batch['response']
        prompt = self.tokenizer.batch_decode(prompt_ids)
        response = self.tokenizer.batch_decode(response_ids)
        pattern = r'\n<|im_start|>user\n(.*?)<|im_end|>'
        for i in range(len(prompt)):
            try:
                matches = re.findall(pattern, prompt[i], re.DOTALL)
                matches = [match.strip() for match in matches if match.strip()]
                prompt[i] = matches[0]
            except:
                continue
        pattern = r'(.*?)<|im_end|>'
        for i in range(len(response)):
            try:
                matches = re.findall(pattern, response[i], re.DOTALL)
                matches = [match.strip() for match in matches if match.strip()]
                response[i] = matches[0]
            except:
                continue
        chosen_margin_reward = []
        rejected_margin_reward = []
        for i in range(len(prompt)):
            if i % 2 == 0:
                messages = [{"role": "user", "content": prompt[i]},
                    {"role": "assistant", "content": response[i]}]
                input_ids = self.reward_tokenizer_modpo.apply_chat_template(messages,return_tensors="pt").to(self.accelerator.device)
                print("padding_token: "+str(self.reward_tokenizer_modpo.pad_token_id))
                print("input_ids: "+str(input_ids))
                print("embed_token.size: "+str(self.reward_model.model.embed_tokens))
                with torch.no_grad():
                    output = self.reward_model(input_ids)
                    score = output.score.float().item()
                
                
                chosen_margin_reward.append(score)

                # try:
                #     chosen_margin_reward.append(self.reward_model([{"role": "user", "content": str(prompt[i])}, {"role": "assistant", "content": str(response[i])}])['score'])
                #     print(prompt[i])
                #     print(response[i])
                #     print("succeed")
                # except:
                #     print("failed!")
                #     print(prompt[i])
                #     print(response[i])
                #     chosen_margin_reward.append(0.0)
            else:
                messages = [{"role": "user", "content": prompt[i]},
                    {"role": "assistant", "content": response[i]}]
                input_ids = self.reward_tokenizer_modpo.apply_chat_template(messages,return_tensors="pt").to(self.accelerator.device)
                with torch.no_grad():
                    output = self.reward_model(input_ids)
                    score = output.score.float().item()
                rejected_margin_reward.append(score)
                # try:
                #     rejected_margin_reward.append(self.reward_model([{"role": "user", "content": str(prompt[i])}, {"role": "assistant", "content": str(response[i])}])['score'])
                #     print(prompt[i])
                #     print(response[i])
                #     print("succeed")
                # except:
                #     print("failed!")
                #     print(prompt[i])
                #     print(response[i])
                #     rejected_margin_reward.append(0.0)
        chosen_margin_reward = torch.tensor(chosen_margin_reward)
        rejected_margin_reward = torch.tensor(rejected_margin_reward)
        return chosen_margin_reward.detach().clone(), rejected_margin_reward.detach().clone()


    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        chosen_margin_reward: torch.FloatTensor, 
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.modpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, chosen_margin_reward, rejected_margin_reward
            )


        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error
        # pdb.set_trace()
        # # 普通 batch 无prompt等信息
        # all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        # modpo batch
        all_logits: "torch.Tensor" = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch['labels'], return_dict=True, use_cache=False).logits.to(torch.float32)

        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        # print(batch_size)
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)

        # chosen_margin_reward, rejected_margin_reward = self.compute_reward_model_score(model,batch)
        # get margin_reward
        batch_size = batch["reward_score"].size(0) // 2
        chosen_reward_score, rejected_reward_score = batch["reward_score"].split(batch_size, dim=0)
        chosen_margin_reward_list = self.tokenizer.batch_decode(chosen_reward_score, skip_special_tokens=True)
        chosen_margin_reward = [float(num) for num in chosen_margin_reward_list]
        chosen_margin_reward = torch.tensor(chosen_margin_reward, dtype=torch.float32)
        rejected_margin_reward_list = self.tokenizer.batch_decode(rejected_reward_score, skip_special_tokens=True)
        rejected_margin_reward = [float(num) for num in rejected_margin_reward_list]
        rejected_margin_reward = torch.tensor(rejected_margin_reward, dtype=torch.float32)

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_margin_reward, 
            rejected_margin_reward
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics
