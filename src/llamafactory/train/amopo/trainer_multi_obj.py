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

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps, get_all_tokens_batch_logps

import pdb

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomMOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
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

    def single_dimension_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes single dimension loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        single_dimension_loss = -F.logsigmoid(self.beta * logits)
        return single_dimension_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.single_dimension_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
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
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)

        # per_token_logits 就是 每个token的probability（0-1之间）， 没有过log（负的）的
        all_logps, valid_length, per_token_logits, loss_mask = get_all_tokens_batch_logps(logits=all_logits, labels=batch["labels"])
        
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 8  # 每个数据有8个input_ids (chosen, rejected) * (no_object, helpfulness, correctness, instruction_following) ,并且是成对出现(chosen_no_obj, rejected_obj, ......)
        
        new_all_logps = all_logps.split(batch_size, dim=0)
        new_all_logits = all_logits.split(batch_size, dim=0)
        new_valid_length= valid_length.split(batch_size, dim=0)
        
        chosen_all_logps = [new_all_logps[i] for i in range(len(new_all_logps)) if i % 2 == 0]
        rejected_all_logps = [new_all_logps[i] for i in range(len(new_all_logps)) if i % 2 != 0]
        chosen_all_logits = [new_all_logits[i] for i in range(len(new_all_logits)) if i % 2 == 0]
        rejected_all_logits = [new_all_logits[i] for i in range(len(new_all_logits)) if i % 2 != 0]
        chosen_all_valid_length = [new_valid_length[i] for i in range(len(new_valid_length)) if i % 2 == 0]
        rejected_all_valid_length = [new_valid_length[i] for i in range(len(new_valid_length)) if i % 2 != 0]


        # 为了系数矩阵
        new_per_token_logits = per_token_logits.split(batch_size, dim=0)
        new_loss_mask = loss_mask.split(batch_size, dim=0)

        chosen_per_token_logits = [new_per_token_logits[i] for i in range(len(new_per_token_logits)) if i % 2 == 0]
        reject_per_token_logits = [new_per_token_logits[i] for i in range(len(new_per_token_logits)) if i % 2 != 0]
        chosen_loss_mask = [new_loss_mask[i] for i in range(len(new_loss_mask)) if i % 2 == 0]
        rejected_loss_mask = [new_loss_mask[i] for i in range(len(new_loss_mask)) if i % 2 != 0]
       
        return chosen_all_logps, rejected_all_logps, chosen_all_logits, rejected_all_logits, chosen_all_valid_length, rejected_all_valid_length, chosen_per_token_logits, reject_per_token_logits, chosen_loss_mask, rejected_loss_mask

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
            reference_all_chosen_logps, reference_all_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_all_chosen_logps, reference_all_rejected_logps

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """

        coefficient_matrix = torch.tensor([[1.0], [1.0], [1.0], [1.0]]).to(self.accelerator.device)
        metrics = {}
        (
            policy_chosen_all_logps, 
            policy_rejected_all_logps, 
            policy_chosen_all_logits, 
            policy_rejected_all_logits, 
            policy_chosen_all_valid_length, 
            policy_rejected_all_valid_length,
            policy_chosen_per_token_logits,
            polocy_rejected_per_token_logits,
            policy_chosen_loss_mask,
            policy_rejected_loss_mask,
        ) = self.concatenated_forward(model, batch)
            
    
        policy_chosen_all_logps = torch.cat(policy_chosen_all_logps)
        policy_rejected_all_logps =torch.cat(policy_rejected_all_logps) 
        policy_chosen_all_logits =torch.cat(policy_chosen_all_logits)
        policy_rejected_all_logits =torch.cat(policy_rejected_all_logits) 
        policy_chosen_all_valid_length =torch.cat(policy_chosen_all_valid_length)
        policy_rejected_all_valid_length =torch.cat(policy_rejected_all_valid_length)

        reference_all_chosen_logps, reference_all_rejected_logps = self.compute_reference_log_probs(model, batch)
        all_losses = []
        all_chosen_rewards = []
        all_rejected_rewards = []
        for i in range(len(policy_chosen_all_logps)):
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_all_logps[i],
                policy_rejected_all_logps[i],
                reference_all_chosen_logps,
                reference_all_rejected_logps,
            )
            all_losses.append(losses)
            all_chosen_rewards.append(chosen_rewards)
            all_rejected_rewards.append(rejected_rewards)

        losses = torch.stack(all_losses)

        # Gaussian distribution for every token
        policy_chosen_per_token_logits = torch.cat(policy_chosen_per_token_logits)
        polocy_rejected_per_token_logits = torch.cat(polocy_rejected_per_token_logits)
        policy_chosen_loss_mask = torch.cat(policy_chosen_loss_mask)
        policy_rejected_loss_mask = torch.cat(policy_rejected_loss_mask)

        policy_chosen_per_token_probs = torch.where(policy_chosen_loss_mask, policy_chosen_per_token_logits, policy_chosen_per_token_logits)
        policy_rejected_per_token_probs = torch.where(policy_rejected_loss_mask, polocy_rejected_per_token_logits, polocy_rejected_per_token_logits)

        policy_chosen_topk_token_probs = policy_chosen_per_token_probs*policy_chosen_loss_mask
        policy_chosen_topk_token_probs_weighted = []
        for probs in policy_chosen_topk_token_probs:
            non_zero_probs = probs[probs != 0]
            # non_zero_probs 为空
            if non_zero_probs.numel() == 0: 
                print('-------policy_chosen_topk_token_probs ---  non_zero_probs 为空-------------')
                policy_chosen_topk_token_probs_weighted.append(0.0)
                continue
            mean = torch.mean(non_zero_probs)
            std_dev = torch.std(non_zero_probs)
            weighted = torch.normal(mean=mean, std=std_dev)
            policy_chosen_topk_token_probs_weighted.append(weighted.item())

        policy_chosen_topk_token_probs_weighted = torch.tensor(policy_chosen_topk_token_probs_weighted).to(self.accelerator.device)

        policy_rejected_topk_token_probs = policy_rejected_per_token_probs*policy_rejected_loss_mask
        policy_rejected_topk_token_probs_weighted = []
        for probs in policy_rejected_topk_token_probs:
            non_zero_probs = probs[probs != 0]
            # non_zero_probs 为空
            if non_zero_probs.numel() == 0: 
                print('-------policy_rejected_topk_token_probs ---  non_zero_probs 为空-------------')
                policy_rejected_topk_token_probs_weighted.append(0.0)
                continue
            mean = torch.mean(non_zero_probs)
            std_dev = torch.std(non_zero_probs)
            weighted = torch.normal(mean=mean, std=std_dev)
            policy_rejected_topk_token_probs_weighted.append(weighted.item())
        policy_rejected_topk_token_probs_weighted = torch.tensor(policy_rejected_topk_token_probs_weighted).to(self.accelerator.device)

        softmax_tensor = torch.add(policy_chosen_topk_token_probs_weighted, policy_rejected_topk_token_probs_weighted)
        
        if torch.isinf(softmax_tensor).any() or torch.isnan(softmax_tensor).any():
            print('-------inf or nan-----::::', softmax_tensor) # check why goto this branch？
            softmax_tensor = torch.where(torch.isinf(softmax_tensor) | torch.isnan(softmax_tensor), torch.tensor(0.0, device=softmax_tensor.device), softmax_tensor)

        softmax_tensor = F.softmax(softmax_tensor, dim=0)
        print("Gaussian Sample weighted assignment: " + str(softmax_tensor))


        losses = losses*softmax_tensor
        all_policy_chosen_logps_avg = policy_chosen_all_logps / policy_chosen_all_valid_length
        sft_loss = - all_policy_chosen_logps_avg.mean()
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # pdb.set_trace()
        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/dpo_margins".format(prefix)] = (all_chosen_rewards[0] - all_rejected_rewards[0]).mean().cpu()
        metrics["{}rewards/helpfulness_margins".format(prefix)] = (all_chosen_rewards[1] - all_rejected_rewards[1]).mean().cpu()
        metrics["{}rewards/correctness_margins".format(prefix)] = (all_chosen_rewards[2] - all_rejected_rewards[2]).mean().cpu()
        metrics["{}rewards/instruction_following_margins".format(prefix)] = (all_chosen_rewards[3] - all_rejected_rewards[3]).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_all_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_all_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_all_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_all_logits.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()
        # pdb.set_trace()
        return losses.mean(), metrics
