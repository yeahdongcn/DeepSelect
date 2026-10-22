import torch, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from deep_select.deep_select_musa import topk as mtopk

def ref(x, k):
    v, i = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    return v, i

def run(x, k, **kw):
    n = x.shape[0]
    ov = torch.empty((n, k), dtype=x.dtype)
    oi = torch.empty((n, k), dtype=kw.get("indices_type", torch.int32))
    mtopk(x, k, None, kw.get("end"), kw.get("sorted", True), kw.get("sorted_index", False),
          ov, oi, kw.get("off"), kw.get("fillidx", 2147483647), float("-inf"),
          kw.get("return_value", True), False)
    return ov, oi

torch.manual_seed(0)
ok = True

# 1. plain agreement with torch.topk
x = torch.randn(6, 512, dtype=torch.bfloat16)
v, i = run(x, 16)
rv, ri = ref(x, 16)
print("1 values equal:", torch.equal(v, rv), "indices equal:", torch.equal(i.to(torch.int64), ri))
ok &= torch.equal(v, rv) and torch.equal(i.to(torch.int64), ri)

# 2. end narrows rows below topk -> fill
x2 = torch.randn(3, 64, dtype=torch.bfloat16)
end = torch.tensor([64, 5, 0], dtype=torch.int32)
v2, i2 = run(x2, 8, end=end)
real_head = torch.equal(i2[1, :5].to(torch.int64), ref(x2[1:2, :5], 5)[1][0])
tail_fill = bool((i2[1, 5:] == 2147483647).all())
empty_fill = bool((i2[2] == 2147483647).all())
print("2 short row: 5 real head", real_head, "| 3 filled tail", tail_fill, "| empty row all fill", empty_fill,
      "| long row ok:", torch.equal(i2[0].to(torch.int64), ref(x2[:1], 8)[1][0]))
ok &= real_head and tail_fill and empty_fill

# 3. output_idx_offset applies, but not to padding
v3, i3 = run(x2, 8, end=end, off=torch.tensor([100, 100, 100], dtype=torch.int32))
expected_off = ref(x2[1:2, :5], 5)[1][0].to(torch.int64) + 100
print("3 offset exact:", torch.equal(i3[1, :5].to(torch.int64), expected_off), "| filled tail untouched:", bool((i3[1, 5:] == 2147483647).all()))
ok &= torch.equal(i3[1, :5].to(torch.int64), expected_off) and bool((i3[1, 5:] == 2147483647).all())

# 4. sorted_index orders ascending by index
v4, i4 = run(x, 32, sorted_index=True)
print("4 sorted_index ascending:", bool((i4[:, 1:] >= i4[:, :-1]).all()),
      "| value set preserved:", torch.equal(torch.sort(v4, -1).values, torch.sort(ref(x, 32)[0], -1).values))
ok &= bool((i4[:, 1:] >= i4[:, :-1]).all())

# 5. NaN poisons the row when abort_when_nan_found
xn = torch.randn(2, 32, dtype=torch.bfloat16); xn[0, 3] = float("nan")
vn = torch.empty((2, 4), dtype=xn.dtype); ino = torch.empty((2, 4), dtype=torch.int32)
mtopk(xn, 4, None, None, True, False, vn, ino, None, 2147483647, float("-inf"), True, True)
print("5 nan row poisoned:", bool((ino[0] == 0x3F3F3F3F).all()), "| clean row normal:", bool((ino[1] != 0x3F3F3F3F).all()))
ok &= bool((ino[0] == 0x3F3F3F3F).all())

# 6. return_value=False still fills indices
ov6, oi6 = run(x, 8, return_value=False)
print("6 indices only:", torch.equal(oi6.to(torch.int64), ref(x, 8)[1]))
ok &= torch.equal(oi6.to(torch.int64), ref(x, 8)[1])

# 7. topk larger than vocab
x7 = torch.randn(2, 10, dtype=torch.float32)
v7, i7 = run(x7, 16)
print("7 topk>vocab: rows", v7.shape, "| head matches:", torch.equal(i7[:, :10].to(torch.int64), torch.argsort(x7, -1, descending=True)))
ok &= torch.equal(i7[:, :10].to(torch.int64), torch.argsort(x7, -1, descending=True)) and bool((i7[:, 10:] == 2147483647).all())

# 8. A *negative* fill value must still land in the padding, offset-free.
#    Regression: "real entry" used to be `indices < vocab` (the row capacity) instead of
#    `~padded`. DeepInfer passes idx_oob_fill_value=-1, and -1 < vocab, so padding was treated
#    as real, offset by `ks`, and the row came back as `ks-1` in its tail: index 99 for ks=100.
x8 = torch.randn(3, 64, dtype=torch.float32)
end8 = torch.tensor([64, 5, 0], dtype=torch.int32)
v8, i8 = run(x8, 8, end=end8, off=torch.tensor([100, 100, 100], dtype=torch.int32), fillidx=-1)
tail8 = bool((i8[1, 5:] == -1).all())
empty8 = bool((i8[2] == -1).all())
print("8 negative fill: tail is -1 not ks-1:", tail8, "| empty row all -1:", empty8,
      "| head offset:", i8[1, 0].item())
ok &= tail8 and empty8 and int(i8[1, 0]) != 99

# 9. Output order is the documented contract: valid entries first, padding last. With fill=-1 a
#    plain ascending sort would put every padded entry *first*, so a caller that reads the first
#    `min(end, topk)` columns as "selected" would see padding as selected.
for si in (False, True):
    v9, i9 = run(x8, 8, end=end8, off=torch.tensor([100, 100, 100], dtype=torch.int32),
                 fillidx=-1, sorted_index=si)
    real_first = bool((i9[1, :5] != -1).all()) and bool((i9[1, 5:] == -1).all())
    print(f"9 sorted_index={si}: real entries first, padding last:", real_first)
    ok &= real_first

# 10. Values stay paired with their indices after the sorted_index permutation (the value fill
#     mask used to be applied positionally *after* sorting, overwriting real values).
v10, i10 = run(x8, 8, end=end8, off=torch.tensor([100, 100, 100], dtype=torch.int32),
               fillidx=-1, sorted_index=True)
pair_errs = []
for r in range(3):
    for c in range(8):
        idx = int(i10[r, c])
        is_pad = idx == -1
        if is_pad != (float(v10[r, c]) == float("-inf")):
            pair_errs.append((r, c))
        elif not is_pad and float(v10[r, c]) != float(x8[r, idx - 100]):
            pair_errs.append((r, c))
print("10 value/index pairing errors:", pair_errs or "none")
ok &= not pair_errs

# 11. Long-context width: 16385 columns is the measured failing prompt length. Rows are kept
#     small here (the full 8192-row chunk lives in tests/test_musa_longcontext.py) but the
#     column count, which is what grows with context, is the real one.
x11 = torch.randn(64, 16385, dtype=torch.float32)
v11, i11 = run(x11, 512, end=torch.full((64,), 16385, dtype=torch.int32), off=torch.zeros(64, dtype=torch.int32),
               fillidx=-1, sorted_index=True)
rv11, ri11 = ref(x11, 512)
# `sorted_index` permutes by index, so values are compared as a multiset and then checked
# pairwise against the input at their own index.
paired11 = torch.equal(torch.gather(x11, 1, i11.to(torch.int64)), v11)
print("11 long-context: index set matches:", torch.equal(i11.to(torch.int64), torch.sort(ri11, -1).values),
      "| value multiset matches:", torch.equal(torch.sort(v11, -1).values, torch.sort(rv11, -1).values),
      "| value/index paired:", paired11)
ok &= (torch.equal(i11.to(torch.int64), torch.sort(ri11, -1).values)
       and torch.equal(torch.sort(v11, -1).values, torch.sort(rv11, -1).values)
       and paired11)

print("RESULT:", "ALL PASS" if ok else "FAILURES")
