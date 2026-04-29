from __future__ import annotations

from typing import Tuple

import torch

RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    sorted_indices_to_remove[0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(indices_to_remove, float("-Inf"), logits)
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def apply_repetition_penalty(
    logits: torch.Tensor, previous_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.
    Based on the scheme: if logit < 0: logit *= penalty else: logit /= penalty
    Works for logits of shape [vocab], [B, vocab], [B, T, vocab].
    """
    if penalty == 1.0 or previous_tokens.numel() == 0:
        return logits

    vocab_size = logits.shape[-1]
    unique_tokens = torch.unique(previous_tokens.to(torch.long))
    unique_tokens = unique_tokens[(unique_tokens >= 0) & (unique_tokens < vocab_size)]

    if unique_tokens.numel() == 0:
        return logits

    logits = logits.clone()
    scores = logits[..., unique_tokens]
    # Apply penalty: if score > 0, divide by penalty. If score <= 0, multiply by penalty.
    # Note: Using torch.where to avoid in-place issues and maintain vectorization.
    new_scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[..., unique_tokens] = new_scores
    return logits


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs
