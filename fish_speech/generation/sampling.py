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
    logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    penalty: float,
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits in a compile-safe way.
    Works for logits of shape [..., vocab].
    """
    if penalty == 1.0:
        return logits

    vocab_size = logits.shape[-1]
    ids = previous_tokens.to(torch.long).reshape(-1)

    # Filter out-of-range tokens in a compile-safe way
    valid_ids_mask = (ids >= 0) & (ids < vocab_size)
    safe_ids = torch.where(valid_ids_mask, ids, torch.zeros_like(ids))

    token_mask = torch.zeros((vocab_size,), dtype=torch.bool, device=logits.device)
    token_mask = token_mask.scatter(0, safe_ids, valid_ids_mask)

    if valid_token_mask is not None:
        token_mask = token_mask & valid_token_mask

    penalized_logits = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(token_mask, penalized_logits, logits)


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
