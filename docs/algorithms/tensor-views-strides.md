---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, views, strides]
workstream: [W9]
module: "[[tensor]]"
summary: "views correct through `At/SetAt` only; `Copy`, same-rank `Reshape` of non-contiguous…"
source: [x/math/tensor/eager/tensor.go, x/math/tensor/eager/linalg.go, x/math/tensor/eager/conv.go, x/math/tensor/types/tensor.go, x/math/tensor/types/core.go]
tests: [x/math/tensor/eager/views_test.go, x/math/tensor/eager/views_zk_test.go, x/math/tensor/eager/slice_test.go, x/math/tensor/eager/strides_test.go, x/math/tensor/eager/tensor_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → view tests pass; probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): Copy/Reshape/Elements/Clone wrong or wasteful on views; SetAt/At panic for INT32/INT64/INT/UINT8"
---
# Tensor strides, offset views, Slice/Reshape/Transpose/Permute/Pad, Copy/Clone, element access

**What it does:** `eager.Tensor{shape, data any, strides []int, offset int}` lets the same backing buffer be seen through several views.
- `strides==nil` means contiguous row-major.
- `Slice`, `Permute`/`Transpose` and some `Reshape` calls return zero-copy views.
- `Pad`, `Unpad`, `BroadcastTo` and `Clone` copy.
- `At`/`SetAt`/`Elements` give float64 element access.

**Where:** `x/math/tensor/eager/tensor.go`:
- `New` 32-39, `NewAs` 42-44, `FromArray` 56-62, `Data` 103-111, `Release` 124-137, `Strides` 166-188, `IsContiguous` 194-206, `DataWithOffset` 219-268
- `Clone` 272-293, `Copy` 300-326
- `getElementAtIndex` 339-371, `setElementAtIndex` 375-407, `At` 412-446, `SetAt` 451-487
- `Reshape` 493-586, `Slice` 592-677, `Elements` 711-747

Elsewhere: `Transpose` 280-309 and `Permute` 314-434 in linalg.go; `Unpad` 1928-2010 and `Pad`/`PadTo` 2017-2098 in conv.go. Interfaces live in `x/math/tensor/types/{tensor,core,manipulation}.go`.

**Inputs → outputs:**
- `Slice(dst, dim, start, len)` returns a view with `offset += start*stride[dim]`, and always stores strides.
- `Permute(nil, dims)` returns a view with permuted strides; strides are stored only if non-canonical.
- `Reshape(nil, shape)` returns a view, except a rank-changing reshape of a non-contiguous tensor, which copies.
- `Pad(dst, [before0,after0,…], value)` is a constant pad that copies.
- `At(i...)` returns float64; a single index on rank>1 is a raw linear index.

## Completeness review

- Complete: no.
  - Element access supports only FP32/FP64/INT16/INT8. `getElementAtIndex` and `setElementAtIndex` hit `default: panic("unsupported data type")` for INT32, INT64, INT and UINT8 (tensor.go:367-369, 403-405). The probe reproduced the panic for all four on both `SetAt` and the getter. This confirms the IOVision report (`tmp/easyrobot/io/graphprobe`).
  - `Slice` with a dst handles only 4 dtypes (tensor.go:660-669).
  - There is no contiguous-ify op (`Contiguous()`), no stride-0 broadcast view, and no `Squeeze`/`Unsqueeze`/`Concat`/`Split`/`Gather` in eager.
- Tested: the views, slice, strides and tensor tests (`TestSlice_ZeroCopyView*`, `TestPermute_ZeroCopyView*`, `TestReshape_ZeroCopyView`, `TestNestedViews`, `Test3D*`, `TestAt_WithOffset`, `TestSetAt_WithOffset`, `TestClone_StridesAndOffset`, `TestViewMemorySharing`, `TestDataWithOffset*`, `TestTensorReleaseReturnsBuffer`, …) check values through `At`/`SetAt`, which do honour offset and strides. All pass (2026-11-09). No test feeds a view into a compute op.
- Correct: **no**. Probes on `big=[4,3] seq`, `v2=big.Slice(nil,1,1,2)` (shape [4,2], strides [3,1]):

  | Operation | Location | Result | Correct |
  |---|---|---|---|
  | `Copy(src view)` | tensor.go:317-323 | row0 [1 2]; uses `Data()` without offset for both src and dst | [2 3] |
  | `Reshape(nil,[2,4])` same rank, non-contiguous | tensor.go:504-511 | row0 [2 3 4 5]; keeps old strides on the new shape | [2 3 5 6] |
  | `Reshape(dst, …)` | tensor.go:576-584 | copies the whole backing slice raw; panics on length mismatch for any view | — |
  | `Elements()` | tensor.go:736-738 | yields 6 elements; skips `linearIdx >= Size()` | 8 |
  | `At(1)` single index | tensor.go:425-431 | storage-order element (offset+1), not logical order | logical order |
  | `Clone()` of a 4-element view | tensor.go:278-292 | copies the full backing buffer (len 8) and keeps offset/strides | 4 elements |
  | `NewAs(t)` | tensor.go:42-44 | aliases `t.Data()` instead of allocating (see [[layer-norm]]) | new buffer |
  | `Unpad`/`PadTo` | conv.go:1990-1999, 2073-2088 | use `Data()` without offset | honour offset |

  - Clone's full-buffer copy is correct but wasteful, and the clone keeps a non-zero offset, so `Release()` never returns it to the pool.
  - Every compute kernel ignores `offset`; see [[tensor-elementwise-broadcast-reduce]] and [[tensor-matmul]].
  - Ownership: views share `data`. `Release()` skips views (offset≠0 or non-contiguous), but a contiguous offset-0 view, such as a `Reshape` of a fresh tensor, can be released while the original still uses the buffer [INFERENCE from tensor.go:124-137].

### Which eager ops honour view offset and strides

`Data()` (tensor.go:103-111) returns the whole backing slice and ignores `offset`. Only code that adds `t.offset` itself, or calls `DataWithOffset()`, is offset-safe. Verified by reading the code; the "probe" rows were also run.

| Honours | Ops (file:line) |
|---|---|
| offset **and** strides | `At`/`SetAt`/`Element.Get/Set` (tensor.go:345, 381, 441-445); `Slice` view and dst copy (tensor.go:630, 654-674); `Permute`/`Transpose` view creation, which keeps offset (linalg.go:392-397); `Reshape` rank-change copy of a non-contiguous tensor (tensor.go:556-562) |
| strides only (offset dropped) | binary arithmetic `Add`/`Subtract`/`Multiply`/`Divide` general path (math.go:17-198); scalar ops (math.go:203-349); unary `Square`…`Negative` (math.go:1081-2368); comparisons and `Where` (math.go:847-2172); reductions `Sum`/`Mean`/`Max`/`Min`/`ArgMax`/`ArgMin` (math.go:427-800) — probe `Sum` gave 10, correct 26; `BroadcastTo` (math.go:400-413); `Permute(dst)` copy (linalg.go:409-427); `Copy` (tensor.go:317-323); `Pad`/`Unpad` (conv.go:2080, 1994); `ReLUGrad`/`SigmoidGrad`/`TanhGrad` stride path (activations.go:115-264); `DropoutForward` (activations.go:396-440) |
| neither (flat contiguous from index 0) | contiguous fast paths of `Add`/`Subtract`/`ScalarMul` (math.go:49-56, 102-109, 225-229); `MatMul` all paths, `MatMulTransposed`, `MatVecMulTransposed` (linalg.go:65-66, 137-138, 184-185, 219-220; linalg_helpers.go:56-58, 199-201) — probe errors 72/16; `Dot`/`Norm` (linalg.go:441-555) — probe 14 vs 77; `AddScaled` (linalg_helpers.go:339-404); `ReLU`/`Sigmoid`/`Tanh` (activations.go:13-111) — probe ReLU gave [1 2 3 4], correct [5 6 7 8]; `Softmax`/`SoftmaxGrad` (activations.go:269-391); every conv/pool/im2col/scatter op (conv.go: `tData := GetTensorData(t)` at 73, 240, 392, 551, 661, 734, 861, 950, 1034, 1099, 1228, 1326, 1419, 1512, 1602, 1668, 1746, 1839); all normalizations, forward and grad (normalizations.go:35, 111, 149, 213, 251, 305, 346, 414, 455, 523); `Reshape(dst)` (tensor.go:576-584); `Clone` (copies the whole buffer and keeps offset, so values are right but it is wasteful) |

So any `Slice` view with offset ≠ 0 (LSTM gate slices, per-head attention slices, KV-cache windows) gives wrong results in every compute op unless it is first copied with `Slice(dst,…)` or a rank-changing `Reshape`.

Transpose/permute views (offset 0) are also wrong in the element-wise ops of the "strides only" row: binary, scalar, unary, comparison and Where. The reason is that `x/iter/st` rank-2 strided apply drops the inner stride (x/iter/st/apply_tensor.go:53-57, apply_matrix.go:21-25). Probe p3: transposed `Add` gave [1 2 3 / 2 3 4]; the correct result is [1 3 5 / 2 4 6]. Reductions (AdvanceOffsets) are correct on transposes.

## Embedded suitability

- `data any` plus type switches in every accessor. `At`/`SetAt` go through float64 (soft-float on ESP32).
- `Slice` allocates a strides slice per view (tensor.go:637-638); `Permute` allocates `used []bool` and new strides (linalg.go:331, 386).
- `Elements` builds an iterator closure. `ID()` uses `unsafe.Pointer`.

## TODOs

- [ ] Add INT32/INT64/INT/UINT8 (and other DataElementType) cases to get/setElementAtIndex, or make them generic — x/math/tensor/eager/tensor.go:339-407; probe panics #p1 #ER-020
- [ ] Make Copy/Unpad/PadTo/Reshape(dst) honour src/dst offset (DataWithOffset) and strides — tensor.go:317-323,576-584; conv.go:1990-1999,2073-2088 #p1 #ER-020
- [ ] Reshape of a non-contiguous tensor with the same rank must copy (or validate stride compatibility) instead of keeping old strides — tensor.go:504-511 #p1 #ER-020
- [ ] Elements(): iterate logical indices without the `linearIdx >= size` skip — tensor.go:736-738 #p1 #ER-020
- [ ] At/SetAt with a single index on rank>1: use logical (row-major) order, not storage order — tensor.go:425-431,463-471 #p2
- [ ] Clone should compact views to a contiguous offset-0 buffer of Size() elements — tensor.go:272-293 #p2
- [ ] NewAs must allocate a new buffer (currently aliases) — tensor.go:42-44 #p1 #ER-020
- [ ] Add `Contiguous()` and stride-0 broadcast views; add tests that pass views into compute ops — tensor.go #p2

## Links

- Module: [[tensor]]
- Related: [[tensor-matmul]], [[tensor-elementwise-broadcast-reduce]], [[type-conversion]], [[iter-workers]]
