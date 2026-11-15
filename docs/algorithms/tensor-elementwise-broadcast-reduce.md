---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, elementwise, reduction]
workstream: [W9]
module: "[[tensor]]"
summary: "all kernels get `Data()` without offset → wrong on `Slice` views (Add/Sum/ReLU probes); no…"
source: [x/math/tensor/eager/math.go, x/math/primitive/fp32/tensor_elementwise.go, x/math/primitive/fp32/tensor_broadcast.go, x/math/primitive/fp32/tensor_reduction.go]
tests: [x/math/tensor/eager/math_test.go, x/math/primitive/fp32/tensor_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → eager 305/305 pass; probe (2026-11-09): ops on offset views read wrong elements"
---
# Eager element-wise ops, broadcasting, reductions, comparisons, Where

**What it does:** Covers the eager tensor arithmetic: binary arithmetic, scalar ops, unary math, comparisons, `Where`, `BroadcastTo`, and the reductions `Sum`/`Mean`/`Max`/`Min`/`ArgMax`/`ArgMin`. The fp32 strided kernels sit in `fp32/tensor_*.go` and the generic iterators in `x/iter`.

**Where:**
- `x/math/tensor/eager/math.go`:
  - Binary: `Add` 17-65, `Subtract` 70-118, `Multiply` 123-158, `Divide` 163-198
  - Scalar: `ScalarMul`/`AddScalar`/`SubScalar`/`DivScalar` 203-349
  - `BroadcastTo` 353-420
  - Reductions: `Sum`/`Mean`/`Max`/`Min` 427-578, `ArgMax` 583-650, `ArgMin` 655-727, `reduceTensor` 733-800
  - `Where` 847-945, `Greater` 949-1027
  - Unary: `Square`…`Pow` 1081-1237, `Abs`/`Sign`/`Cos`/`Sin`/`Negative` 2177-2368
  - Comparisons: `Equal`/`Less`/… 1241-2172
  - `Fill`/`FillFunc` 2373-2564
- fp32 kernels: `ElemAdd`… `tensor_elementwise.go:12-361`, `BroadcastStridesTo`/`ExpandTo` `tensor_broadcast.go:16-92`, `ReduceSum/Mean/Max/Min/Argmax/Argmin` `tensor_reduction.go:8-273`.

**Inputs → outputs:**
- Binary ops need `t.Shape()==other.Shape()`. **There is no implicit broadcasting**; probe `[2,3]+[3]` panicked "shape mismatch". `BroadcastTo` materialises a copy (it is not a stride-0 view).
- nil dst means in-place on `t` for arithmetic/unary. Reductions allocate a new tensor.
- Reductions drop the reduced dims (no keepdims). A full reduction yields shape `[1]`.
- `ArgMax`/`ArgMin` return indices as **float32 in the input dtype** (probe dtype=FP32), so they are exact only below 2²⁴.
- Dtypes:
  - Arithmetic, unary math and reductions are FP32 only and panic otherwise.
  - Comparisons, `Where`, `Sign`, `Negative`, `Fill` dispatch over FP32/FP64/INT32/INT64/INT/INT16/INT8, with no UINT8.
  - `OnesLike`/`FullLike` assert `[]float32` (math.go:1047, 1072), so they panic for other dtypes.

## Completeness review

- Complete: no.
  - No broadcasting in binary ops.
  - No keepdims, `Prod`, `Var`/`Std`, `CumSum` or `TopK`.
  - No elementwise `Maximum`/`Minimum`, `Clip` or `Rsqrt`. Qwen-style RMSNorm/attention built from primitives would need these.
- Tested: math_test.go has value tests for `TestAdd`, `TestSub`, `TestMul`, `TestDiv`, `TestScale`, `TestAddTo`, `TestMulTo`, `TestSum`, `TestMean`, `TestMax`, `TestMin`, `TestArgMax`, `TestWhere`, `TestEqual`, `TestGreater`, `TestLess`, `TestBroadcastTo`, `TestExp`, `TestLog`, `TestSqrt`, `TestPow`, `TestSquare`, `TestAbs`, `TestSign`, `TestCos`, `TestSin`, `TestNegative`. All pass (2026-11-09). No `TestArgMin`. All use offset-0 tensors.
- Correct: **no** on views.
  - Every kernel receives `Data()`, the full backing slice; strides are passed but `offset` never is. So any `Slice` view (offset ≠ 0) reads and writes the wrong elements.
    - Probe on `Slice(nil,0,2,2)` of `[[1,2],[3,4],[5,6],[7,8]]`: `Add(ones)` gave [2 3 4 5] (correct [6 7 8 9]), `Sum` gave 10 (correct 26), `ReLU` gave [1 2 3 4] (correct [5 6 7 8]).
    - **Transposed/permuted views are also wrong for element-wise ops.** `iter.ElemApply*Strided` (default build without `use_mt`) calls `x/iter/st`, whose rank-2 path delegates to `ElemMatApply*Strided` with only the leading strides, so `stride[1]` is dropped (x/iter/st/apply_tensor.go:53-57, apply_matrix.go:21-25; reported by InfraNotes). Rank ≥3 decomposes to that rank-2 path.
      - Probe p3: `Transpose` of [3,2] then `Add(zeros)` gave [1 2 3 / 2 3 4]; the correct result is [1 3 5 / 2 4 6].
      - `Exp(transpose)[0,1]` gave 7.39; the correct value is 20.09.
      - Rank-3 `Permute([0,2,1])` then `Add` gave row0 [1 2 3]; the correct row is [1 3 5].
      - Reductions use `AdvanceOffsets` and are correct on transposes: `Sum(dim1)` gave [9 12].
      - ~24 call sites in fp32/tensor_elementwise.go are affected.
  - IEEE divergences, by design comments but undocumented at the API:
    - `Divide` by 0 leaves dst unchanged (tensor_elementwise.go:34-81); probe dst prefilled 7 stayed 7, where IEEE gives +Inf.
    - `DivScalar(0)` is a silent no-op (tensor_elementwise.go:282-292).
    - `Sqrt(-4)=0` (tensor_elementwise.go:161-168); `Log(-4)=-Inf` (tensor_elementwise.go:184-191). IEEE gives NaN for both.
  - `ReduceMax`/`ReduceMin` initialise to ∓MaxFloat32, not ∓Inf, and skip NaN (tensor_reduction.go:83-86).
  - Contiguous `Add`/`Subtract` fast path uses `Axpy` and ignores offset too (math.go:49-56).

## Embedded suitability

- `reduceTensor` allocates a `map[int]struct{}` per call (math.go:739-742); `ArgMin` allocates `[]int32` per call (math.go:697).
- Each op type-switches on `Data() any`. Kernels use generic `iter.ElemApply*Strided[float32]` with a closure per element. That is an indirect call per element unless inlined, which TinyGo may not do [INFERENCE].
- Scalars arrive as float64 and are converted once. Comparison paths are generic over 7 types, which grows code size.

## TODOs

- [ ] Pass `offset` (use `DataWithOffset()` or slice data at offset) into every fp32/iter kernel call in math.go — x/math/tensor/eager/math.go:38-60,141-152,…; probe Add/Sum/ReLU on Slice view wrong #p1 #ER-020
- [ ] Implement NumPy-style broadcasting in binary ops (stride-0 views via `BroadcastStridesTo`) instead of shape-equality panics — math.go:24-26; tensor_broadcast.go:16 #p2
- [ ] Return ArgMax/ArgMin as INT32/INT64 tensors — math.go:583-727 #p2
- [ ] Make Divide/DivScalar/Sqrt/Log follow IEEE (Inf/NaN) or document clearly — tensor_elementwise.go:34-81,161-191,282-292 #p2
- [ ] Add keepdims, Maximum/Minimum/Clip/Rsqrt, Var/Std reductions needed by transformer blocks — math.go #p2
- [ ] Remove per-call map allocation in reduceTensor (use a [MAX_DIMS]bool mask) — math.go:739-742 #p3
- [ ] Add TestArgMin and tests on offset/strided views — math_test.go #p2

## Links

- Module: [[tensor]]
- Related: [[tensor-views-strides]], [[activation-kernels]], [[iter-workers]], [[type-conversion]]
