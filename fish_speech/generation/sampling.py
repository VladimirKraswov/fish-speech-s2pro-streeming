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
    """
    if penalty == 1.0 or previous_tokens.numel() == 0:
        return logits

    # Gather unique tokens from the window, exclude dummy 0 if possible
    # However, 0 might be a valid semantic token.
    # In decode.py, we'll try to pass only real tokens.
    unique_tokens = torch.unique(previous_tokens)

    # Apply penalty to the logits of previously generated tokens
    score = torch.gather(logits, -1, unique_tokens.unsqueeze(0).expand(logits.shape[0], -1))

    # if score < 0, multiply by penalty, else divide
    # we can do this without explicit if by using torch.where
    score = torch.where(score < 0, score * penalty, score / penalty)

    logits.scatter_(-1, unique_tokens.unsqueeze(0).expand(logits.shape[0], -1), score)
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
