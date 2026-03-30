
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
   """
   pi_logps: policy logprobs, shape (B,)
   ref_logps: reference model logprobs, shape (B,)
   yw_idxs: preferred completion indices in [0, B-1], shape (T,)
   yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
   beta: temperature controlling strength of KL penalty
   Each pair of (yw_idxs[i], yl_idxs[i]) represents the
   indices of a single preference pair.
   """
   pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
   ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
   pi_logratios = pi_yw_logps - pi_yl_logps
   ref_logratios = ref_yw_logps - ref_yl_logps
   losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
   rewards = beta * (pi_logps - ref_logps).detach()
   return losses, rewards


def cal_dpo_loss(chosen_pi_logps, chosen_ref_logps, rejected_pi_logps,
rejected_ref_logps, beta):
   """
   chosen_pi_logps: policy logprobs for the preferred responses, shape (B, )
   chosen_ref_logps: reference logprobs for the preferred responses, shape (B, )
   rejected_pi_logps: policy logprobs for the dispreferred responses, shape (B, )
   rejected_ref_logps: reference logprobs for the dispreferred responses, shape (B, )
   beta: the parameterization coefficient that defines the residual model
   """
   chosen_reward = chosen_pi_logps - chosen_ref_logps
   reject_reward = rejected_pi_logps - rejected_ref_logps
   dpo_losses = -F.logsigmoid(chosen_reward - reject_reward)
   # our method requires a simple modification on DPO with one additional line of code
   cal_losses = F.mse_loss(chosen_reward, 0.5*1/beta) + F.mse_loss(reject_reward, -0.5*1/beta)
   cal_dpo_losses = dpo_losses + cal_losses
   return cal_dpo_losses

def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - self.gamma_beta_ratio

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards
