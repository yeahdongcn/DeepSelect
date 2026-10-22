"""Long-context cases for the MUSA backend -- the length-dependent class.

Why this file exists
--------------------
DeepInfer's long-context prefill (17408 and 16385 tokens) aborted with
``MUSA error: an illegal memory access`` on all eight ranks, and the traceback pointed at
``deep_select_musa.py``'s first device sync, reached from
``index_attn._topk_prefill -> deep_select.interface.topk``. The shapes that reach `topk`
there are not the short test cases: rows are a chunk of the query block (<= 8192) and each
row is the **compressed window**, which grows with the context (16384+ columns).

Two separate things are worth protecting, and they are not the same test:

1. **The call contract at real long-context shapes.** `_topk_prefill` passes
   ``topk=512, end=klen, indices_type=int32, sorted_index=True, output_idx=<a row slice>,
   output_idx_offset=ks, idx_oob_fill_value=-1, return_value=False``. This module builds
   that exact call, over the chunked loop the engine uses, at 16385 and 17408 columns.

2. **The reporting behaviour.** `topk`'s first device read is the NaN guard
   (``bool(bad_rows.any())``), so an *asynchronous* illegal access from the **caller's**
   producer is reported there, not in the caller. That is why the DeepInfer traceback
   pointed into this file. The producer here is deliberately unsynchronised, so the suite
   exercises "my caller's work is still in flight when I start" as a documented property
   rather than as a mystery the next time it shows up in a traceback.

Note on padding at long context: with `klen >= topk` a row has no padding at all, which is
why the padding defects could not have produced the long-context fault. The `klen < topk`
row below covers the padded path at the same width, and `tests/test_musa_backend.py` covers
its offset/fill arithmetic.

What this file does NOT do: it does not inject a fault. A deliberate device fault is not
expressible from Python without a custom kernel, so "an async fault surfaces in the first
sync" is covered as the *ordering* property above, and the fault itself stays a caller-side
reproduction.

Run:
    python tests/test_musa_longcontext.py            # full rows on MUSA, smaller on CPU
    python tests/test_musa_longcontext.py --rows 256 # quicker, same column widths
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from deep_select.deep_select_musa import topk as musa_topk  # noqa: E402

# `_topk_prefill`'s arguments, read from `index_attn.py` and the checkpoint
# (`index_topk: 512` -> `MQAIndexer.topk_tokens`).
TOPK_TOKENS = 512
CHUNK_ROWS = 8192
LONG_WIDTHS = (16385, 17408)      # the two measured failing prompts
NARROW_LEN = 100                  # a row whose window is shorter than topk -> padding


def _device():
    if hasattr(torch.version, "musa") and torch.version.musa is not None:
        return "musa"
    return "cpu"


def _sync(dev):
    if dev == "musa":
        torch.musa.synchronize()


def async_producer(rows, width, dev, seed):
    """Stand in for `_compute_prefill_p_sum`, left in flight on purpose.

    This launches real device work and returns **without synchronising**, which is what the
    engine does between `fp8_fp4_mqa_logits` and `topk`. Any error it raises asynchronously
    is therefore observed at `topk`'s first sync -- the behaviour this file documents.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(rows, width, generator=gen, dtype=torch.float32).to(dev)
    # The engine's logits row is dense, but the candidate pre-filter (layers >= 20) leaves
    # -inf outside the selected blocks; keep a band of it so masks stay exercised.
    if width > 64:
        base[:, -32:] = float("-inf")
    return base


def reference_row(inp, row, k, length=None):
    """Top-k of one row, with the row tail masked the way `end` masks it in the engine."""
    r = inp[row].to(torch.float32).clone()
    if length is not None and length < r.numel():
        r[length:] = float("-inf")
    values, indices = torch.topk(r, k, dim=-1, largest=True, sorted=True)
    return values.cpu(), indices.cpu()


def run_case(rows, width, dev):
    """One chunked pass with the engine's exact call; returns (ok, detail)."""
    total_q = rows
    topk_index = torch.empty(total_q, TOPK_TOKENS, dtype=torch.int32, device=dev)
    # `klen` is the compressed length. Long-context prefill rows are full windows; the last two
    # rows are deliberately short -- one by a single token, one below `topk` -- so the masked
    # tail and the padded path are exercised at the real width.
    klen = torch.full((total_q,), width, dtype=torch.int32, device=dev)
    if total_q >= 2:
        klen[-2] = width - 1
        klen[-1] = NARROW_LEN
    ks = torch.zeros(total_q, dtype=torch.int32, device=dev)

    # Async producer: no synchronize between this and the topk call below.
    p_sum = async_producer(total_q, width, dev, seed=1000 + width)

    for start_q in range(0, total_q, CHUNK_ROWS):
        chunk = slice(start_q, min(start_q + CHUNK_ROWS, total_q))
        musa_topk(
            p_sum[chunk], TOPK_TOKENS,
            None,                     # begin: unsupported by every backend
            klen[chunk],              # end: the compressed length, per row
            False,                    # sorted (output_val order)
            True,                     # sorted_index
            None,                     # output_val: return_value=False
            topk_index[chunk],
            ks[chunk],
            -1,                       # idx_oob_fill_value, as DeepInfer passes it
            float("-inf"),
            False,                    # return_value
            True,                     # abort_when_nan_found: this is the first sync
        )

    _sync(dev)

    for row in (0, total_q - 2, total_q - 1):
        seg = topk_index[row].cpu()
        length = int(klen[row])
        n_real = min(length, TOPK_TOKENS)
        real, pad = seg[:n_real], seg[n_real:]

        # The contract: valid entries first (ascending by index, because sorted_index=True),
        # padding last, and padding is exactly the fill value DeepInfer asks for.
        if not bool((pad == -1).all()):
            bad = pad[pad != -1][:4].tolist()
            return False, f"row {row}: padding not at the tail/fill, e.g. {bad}"
        idx = real.to(torch.int64)
        if bool((idx < 0).any()) or bool((idx >= length).any()):
            return False, f"row {row}: index outside [0,{length})"
        if bool((idx[1:] < idx[:-1]).any()):
            return False, f"row {row}: indices not ascending"
        if len(torch.unique(idx)) != idx.numel():
            return False, f"row {row}: duplicate indices"

        ref_v, ref_i = reference_row(p_sum, row, n_real, length=length)
        # `sorted_index=True` returns the selected set in index order, so the values are a
        # multiset: compare them sorted. `tests/test_musa_backend.py` checks value/index
        # pairing positionally, including through the offset and fill paths.
        got_v = torch.gather(p_sum[row].cpu(), 0, idx.cpu())
        if not torch.equal(torch.sort(got_v).values, torch.sort(ref_v).values):
            worst = (torch.sort(got_v).values - torch.sort(ref_v).values).abs().max().item()
            return False, f"row {row}: value multiset differs from torch.topk, max|d|={worst}"
        if not torch.equal(idx.cpu(), ref_i.sort().values):
            return False, f"row {row}: selected set differs from torch.topk"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=0,
                        help="Rows per case (0 = %d on MUSA, 2048 on CPU)" % CHUNK_ROWS)
    parser.add_argument("--widths", type=int, nargs="+", default=list(LONG_WIDTHS))
    args = parser.parse_args()

    dev = _device()
    rows = args.rows or (CHUNK_ROWS if dev == "musa" else 2048)
    print(f"device={dev} rows={rows} topk={TOPK_TOKENS} widths={args.widths}")

    ok = True
    for width in args.widths:
        case_ok, detail = run_case(rows, width, dev)
        print(f"width={width:6d} -> {'PASS' if case_ok else 'FAIL'} ({detail})")
        ok &= case_ok

    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
