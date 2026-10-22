"""MUSA backend for DeepSelect, implemented with ``torch.topk``.

MUSA is CUDA-compatible at the source level, but the shipped kernels are TMA-based
(``v3``/``v3_cluster``/``v3_fp32`` instantiations), and TMA has no direct MUSA
counterpart — MUSA's equivalent is TME. So a MUSA backend cannot reuse those
instantiations; it needs its own kernel work.

This module exists so that the *contract* is available on MUSA before that work
lands. It is a correctness path, not a performance substitute: it reproduces the
documented semantics (including the out-of-range fill and the ``topk`` larger than
the valid row length case) using ``torch.topk``. The CUDA and Ascend backends are
2-20x faster than this; treat any timing from here as a floor, not a result.

Adding a real MUSA kernel later means replacing ``_topk_impl`` inside this module
while keeping ``topk`` and ``get_alignment_requirement`` as the platform surface,
which is exactly the split the Ascend backend uses (``kernel.asc`` behind the same
two entry points).
"""

from __future__ import annotations

import torch


def get_alignment_requirement():
    """Stride requirements in bytes for (input, output).

    The compiled backends require an alignment so their vectorised loads are valid.
    ``torch.topk`` has no such constraint, but the caller divides this by the element
    size to round the output stride, so it has to be a real byte count: the documented
    contract is 32 Bytes, and anything smaller yields a zero row stride for dtypes
    wider than one byte.
    """
    return (32, 32)


def _gather_keep_order(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, -1, order)


def topk(
    input,
    topk,
    begin,
    end,
    sorted_,
    sorted_index,
    output_val,
    output_idx,
    output_idx_offset,
    idx_oob_fill_value,
    value_oob_fill_value,
    return_value,
    abort_when_nan_found,
):
    """Select the largest ``topk`` entries of each row.

    Mirrors the documented backend contract: results are written in place into
    ``output_val`` and ``output_idx`` and nothing is returned.

    Row ranges are ``[begin[i], end[i])``; ``begin`` is rejected by the caller, so only
    ``end`` can narrow a row. Rows shorter than ``topk`` are padded with
    ``idx_oob_fill_value`` / ``value_oob_fill_value``, and the padding never receives
    ``output_idx_offset``.
    """
    assert begin is None, "`begin` is not supported by this backend, matching the others"

    device = input.device
    n_rows, vocab = input.shape

    if end is None:
        lengths = torch.full((n_rows,), vocab, device=device, dtype=torch.long)
    else:
        lengths = end.to(torch.long).clamp(min=0, max=vocab)

    # NaN policy: the documented behaviour is to poison the row rather than return
    # a plausible-looking wrong index. Surfacing a corrupt row is the point.
    if abort_when_nan_found and input.numel() > 0:
        in_range = torch.arange(vocab, device=device)[None, :] < lengths[:, None]
        bad_rows = torch.isnan(input).logical_and(in_range).any(dim=-1)
        if bool(bad_rows.any()):
            # Only the offending rows are poisoned; a clean row in the same batch must
            # still return a usable answer.
            output_idx[bad_rows] = 0x3F3F3F3F
            if return_value and output_val is not None:
                output_val[bad_rows] = value_oob_fill_value
            if bool(bad_rows.all()):
                return

    k = int(topk)
    width = max(k, vocab)

    work = input
    if width != vocab:
        pad = torch.full((n_rows, width - vocab), float("-inf"), device=device, dtype=input.dtype)
        work = torch.cat((work, pad), dim=-1)

    # Mask the tail of each row so a narrowed row cannot select beyond `end`.
    if end is not None:
        valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
        work = work.masked_fill(~valid, float("-inf"))

    values, indices = torch.topk(work, k, dim=-1, largest=True, sorted=True)

    # Padding is identified per entry, not per row: a row with length 5 and topk 8 keeps
    # its 5 real indices and fills the other 3, and a topk larger than the vocabulary
    # keeps the real head instead of blanking the row.
    padded = indices >= lengths[:, None]
    has_padding = bool(padded.any())
    if has_padding:
        fill_idx = torch.full_like(indices, int(idx_oob_fill_value))
        indices = torch.where(padded, fill_idx, indices)
        if return_value and output_val is not None:
            fill_val = torch.full_like(values, value_oob_fill_value)
            values = torch.where(padded, fill_val, values)

    # "Real entry" is the complement of `padded`, not `indices < vocab`. The latter is the
    # row *capacity*, and the fill value can be smaller than it: DeepInfer passes
    # `idx_oob_fill_value=-1`, so `-1 < vocab` is true and every padded entry would be
    # offset to `output_idx_offset - 1` -- a valid-looking key index instead of the -1 the
    # caller tests for. Keep the predicate that was actually measured.
    in_place = ~padded
    if output_idx_offset is not None:
        offset = output_idx_offset.to(torch.long)[:, None]
        shifted = indices + offset
        # Padding must stay at the fill value, never offset.
        indices = torch.where(in_place, shifted, indices)

    # Output order is part of the contract, and it is not the fill value's numeric order:
    # "valid elements will be gathered at the beginning of values and indices", the rest
    # filled. `sorted_index` sorts that real block ascending. The fill value is chosen by
    # the caller (-1 for DeepInfer, 2147483647 by default), so with -1 a plain ascending
    # sort would put every padded entry *first*; the real block is therefore pinned first
    # by a stable partition, and `values` follows the same permutation so every value stays
    # paired with its index.
    perm = None
    if sorted_index and k > 1:
        perm = torch.argsort(indices.to(torch.int64), dim=-1, stable=True)
    if has_padding:
        padded_sorted = torch.gather(padded.to(torch.int64), -1, perm) if perm is not None else padded.to(torch.int64)
        partition = torch.argsort(padded_sorted, dim=-1, stable=True)
        perm = partition if perm is None else torch.gather(perm, -1, partition)
    if perm is not None:
        indices = torch.gather(indices, -1, perm)
        if return_value and output_val is not None:
            values = _gather_keep_order(values, perm)

    if abort_when_nan_found and input.numel() > 0 and bool(bad_rows.any()):
        keep = ~bad_rows
        output_idx[keep] = indices.to(output_idx.dtype)[keep]
        if return_value and output_val is not None:
            output_val[keep] = values.to(output_val.dtype)[keep]
        return

    output_idx.copy_(indices.to(output_idx.dtype))
    if return_value and output_val is not None:
        output_val.copy_(values.to(output_val.dtype))
