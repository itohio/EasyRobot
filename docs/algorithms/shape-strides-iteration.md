---
type: algorithm
status: tested
tags: [easyrobot, tensor, shape, strides]
workstream: W9
module: "[[tensor]]"
summary: "scalar size inconsistency"
source: [x/math/shape/shape.go, x/math/shape/strides.go, x/math/shape/iter.go]
tests: [x/math/shape/shape_test.go, x/math/shape/helpers_test.go, x/math/shape/iter_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09): x/math/shape 18/18 pass (value-checking tables)"
---
# Shape, strides and offset iteration (x/math/shape)

**What it does:** Row-major shape helpers used by every tensor/iter kernel: `Shape` type (`Rank`, `Size`, `Equal`, `Strides`, `IsContiguous`, `ValidateAxes`, `Clone`, `ToSlice`), free functions `ComputeStrides`, `SizeFromShape`, `EnsureStrides`, `IsContiguous`, and odometer offset walkers `AdvanceOffsets{,3,4}`, `IterateOffsets[WithIndices]`, `ComputeStrideOffset`.

**Where:** `x/math/shape/shape.go:8` `MAX_DIMS = 16`; `:14-98` `Shape` methods; `x/math/shape/strides.go` `ComputeStrides` (last dim stride 1), `SizeFromShape` (0 for empty or any dim ≤0), `EnsureStrides` (recompute if `len(strides)!=rank`), `IsContiguous` (exact equality with canonical strides); `x/math/shape/iter.go` `AdvanceOffsets` (increment last index, add stride, carry: subtract `stride*shape[d]`), variants for 3/4 operands.

**Inputs → outputs:** shapes `[]int` (rank ≤16 when using stack scratch), element strides `[]int`; offsets in elements.

## Completeness review

- Complete: yes for row-major dense/strided views. No negative-stride or broadcasting (stride 0) helpers beyond what `AdvanceOffsets` supports implicitly (stride 0 works arithmetically).
- Tested: 18 tests pass, value-checking (`TestComputeStrides`, `TestEnsureStrides`, `TestIsContiguous`, `TestAdvanceOffsets_Iteration`, `TestIterateOffsetsWithIndices`, `TestComputeStrideOffset`, `TestShape*`).
- Correct: yes within tested scope. Semantics caveats below.

**Defects / caveats**
- `x/math/shape/shape.go:24-31` vs `strides.go` `SizeFromShape`: scalar (rank 0) size is **1** via `Shape.Size()` but **0** via `SizeFromShape` — kernels using the latter skip rank-0 tensors.
- `x/math/shape/shape.go:13-16` `NewShape` doc says "returns a copy" but aliases `dims`; `:81-88` `ToSlice` doc says copy, returns the same backing array.
- `x/math/shape/shape.go:94-97` `Clone` and `strides.go` `ComputeStrides(nil, …)` slice a local `[MAX_DIMS]int` → heap escape (one alloc) and panic for rank > 16.
- `x/math/shape/shape.go:59-79` `ValidateAxes` allocates a map and sorts caller's slice in place.
- `IsContiguous` treats size-1 dims with arbitrary stride as non-contiguous (conservative; causes slow paths only).

**Embedded suitability:** allocation-free except `Clone`/`ComputeStrides(nil)`/`ValidateAxes`; int arithmetic only; no generics, no goroutines. `MAX_DIMS=16` stack arrays (128 B each on 64-bit, 64 B on 32-bit) appear in many kernels — fine for ESP32 stacks but adds up in recursion.

## TODOs

- [ ] Unify scalar size semantics (`SizeFromShape([])` vs `Shape.Size()`) — x/math/shape/shape.go:27, x/math/shape/strides.go #p2
- [ ] Make `NewShape`/`ToSlice` copy or fix docs — x/math/shape/shape.go:14,82 #p3
- [ ] `Clone` should `make` exactly once explicitly and guard rank>MAX_DIMS — x/math/shape/shape.go:94 #p3

## Links

- Module: [[tensor]]
- Related: [[tensor-iterators]], [[numeric-cast-and-clamp]]
