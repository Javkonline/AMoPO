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

from ..trainer_utils import create_custom_optimzer, create_custom_scheduler, get_all_tokens_batch_logps

import pdb

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomAMoPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):

        self.finetuning_args = finetuning_args

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)


    def single_object_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        logits = chosen_logps - rejected_logps
        single_object_loss = -F.logsigmoid(self.beta * logits)
        return single_object_loss

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
        losses = self.single_object_loss(policy_chosen_logps, policy_rejected_logps)
        
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

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

        # per_token_logits: every token probability
        all_logps, valid_length, per_token_logits, loss_mask = get_all_tokens_batch_logps(logits=all_logits, labels=batch["labels"])
        
        all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 8  # 8 input_ids (chosen, rejected) * (no_object, helpfulness, correctness, instruction_following) ,with paired (chosen_no_obj, rejected_obj, ......)
        
        new_all_logps = all_logps.split(batch_size, dim=0)
        new_all_logits = all_logits.split(batch_size, dim=0)
        new_valid_length= valid_length.split(batch_size, dim=0)
        
        chosen_all_logps = [new_all_logps[i] for i in range(len(new_all_logps)) if i % 2 == 0]
        rejected_all_logps = [new_all_logps[i] for i in range(len(new_all_logps)) if i % 2 != 0]
        chosen_all_logits = [new_all_logits[i] for i in range(len(new_all_logits)) if i % 2 == 0]
        rejected_all_logits = [new_all_logits[i] for i in range(len(new_all_logits)) if i % 2 != 0]
        chosen_all_valid_length = [new_valid_length[i] for i in range(len(new_valid_length)) if i % 2 == 0]
        rejected_all_valid_length = [new_valid_length[i] for i in range(len(new_valid_length)) if i % 2 != 0]


        # for adaptive weight assignment
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

        #  对每个分布的token进行高斯分布采样
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

        losses = losses*softmax_tensor

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
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
        return losses.mean(), metrics
