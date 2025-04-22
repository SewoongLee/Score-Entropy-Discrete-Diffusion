from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import torch
import torch.nn.functional as F

def dpo_loss(model, input_ids, sigma, chosen_ids, rejected_ids, beta=1.0):
    """
    Compute DPO loss given model, inputs, sigma, and two outputs: chosen and rejected.
    """

    def compute_logprobs(output_logits, target_ids):
        log_probs = F.log_softmax(output_logits, dim=-1)   # [B, T, V]
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return target_log_probs.sum(dim=-1)  # [B]

    # Forward pass for chosen and rejected outputs
    logits_chosen = model.forward(input_ids, sigma)         # [B, T, V]
    logits_rejected = model.forward(input_ids, sigma)       # [B, T, V]

    logp_chosen = compute_logprobs(logits_chosen, chosen_ids)     # [B]
    logp_rejected = compute_logprobs(logits_rejected, rejected_ids)  # [B]

    # DPO loss
    diff = beta * (logp_chosen - logp_rejected)
    loss = -F.logsigmoid(diff)  # equivalent to cross entropy with target=1
    return loss.mean()
