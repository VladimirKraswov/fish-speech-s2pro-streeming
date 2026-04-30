from __future__ import annotations

from typing import Tuple

import torch

RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    if 0 < top_k < vocab_size:
        k = min(int(top_k), vocab_size)
        top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
        top_vals, order = torch.sort(top_vals, descending=True, dim=-1)
        top_idx = top_idx.gather(dim=-1, index=order)

        top_vals = top_vals / torch.clip(temperature, min=1e-5)
        top_probs = torch.nn.functional.softmax(top_vals, dim=-1)
        cum_probs = torch.cumsum(top_probs, dim=-1)
        remove = cum_probs > top_p
        remove[..., 0] = False
        top_vals = torch.where(remove, float("-Inf"), top_vals)
        top_probs = torch.nn.functional.softmax(top_vals, dim=-1)

        probs = torch.zeros_like(logits)
        return probs.scatter(dim=-1, index=top_idx, src=top_probs)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k if top_k > 0 else torch.zeros_like(indices, dtype=torch.bool)
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
    ids = previous_tokens.to(device=logits.device, dtype=torch.long).reshape(-1)

    # Filter out-of-range tokens in a compile-safe way
    valid_ids_mask = (ids >= 0) & (ids < vocab_size)
    invalid_bucket = torch.full_like(ids, vocab_size)
    safe_ids = torch.where(valid_ids_mask, ids, invalid_bucket)

    token_mask = torch.zeros((vocab_size + 1,), dtype=torch.bool, device=logits.device)
    token_mask = token_mask.scatter(0, safe_ids, valid_ids_mask)
    token_mask = token_mask[:vocab_size]

    if valid_token_mask is not None:
        valid_token_mask = valid_token_mask.to(device=logits.device, dtype=torch.bool)
        valid_token_mask = valid_token_mask.reshape(-1)
        if valid_token_mask.shape[0] != vocab_size:
            raise ValueError(
                "valid_token_mask must have shape [vocab_size], "
                f"got {tuple(valid_token_mask.shape)} for vocab_size={vocab_size}"
            )
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
