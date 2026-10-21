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

print("RESULT:", "ALL PASS" if ok else "FAILURES")
