---
type: algorithm
status: partial
tags: [easyrobot, iter, tensor, strided]
workstream: W9
module: "[[iter-workers]]"
summary: "rank ≥ 2 strided apply ignores the inner stride (`x/iter/st/apply_tensor.go:53-57`…"
source: [x/iter/apply_tensor_st.go, x/iter/apply_tensor_mt.go, x/iter/iterators_st.go, x/iter/iterators_mt.go, x/iter/comparison_st.go, x/iter/scalar.go, x/iter/unary.go, x/iter/ternary.go, x/iter/comparison_vector.go, x/iter/st/apply_tensor.go, x/iter/st/apply_matrix.go, x/iter/st/apply_vector.go, x/iter/st/comparison.go, x/iter/st/iterators.go, x/iter/mt/apply_tensor.go, x/iter/mt/iterators.go, x/iter/ANALYSIS.md]
tests: [x/iter/apply_test.go, x/iter/comparison_test.go, x/iter/iterators_test.go, x/iter/scalar_test.go, x/iter/unary_test.go, x/iter/ternary_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/iter 57/57 pass; x/iter/st, x/iter/mt no test files; go build -tags use_mt ./x/iter fails (import cycle); probe InfraNotes/iterprobe: transposed strided apply wrong"
---
# Tensor iterators and element-wise apply (x/iter)

**What it does:** Generic element-wise kernels over flat slices with shape/strides: `ElemApply{Unary,Binary,Ternary}[Scalar][Strided]`, comparisons (`ElemGreaterThan`, `ElemEqual`, … → 0/1 in dst), `ElemFill`, `ElemSign`, `ElemNegative`, `ElemWhere`, plus range-over-func index iterators (`Elements`, `ElementsStrided`, `ElementsVec/Mat[Strided]`, `ElementsWindow(s)`, `ElementsIndices[Strided]`). The fp32 primitives and eager tensor ops are built on these (24 call sites in `x/math/primitive/fp32/tensor_elementwise.go`).

**Where:**
- Front package `x/iter` re-exports `st` (`//go:build !use_mt`: `apply_*_st.go`, `comparison_st.go`, `iterators_st.go`) or `mt` (`use_mt`: `*_mt.go`).
- `x/iter/st/apply_tensor.go:10-19` contiguous binary; `:23-89` strided: contiguous → `ElemVecApply…` (`:41-45`), rank 1 → vec with stride, rank 2 → `ElemMatApply…Strided(…, ld=strides[0])` (`:53-57`), rank ≥3 → recursion on dim 0 (`:58-88`). Same pattern for unary/ternary/scalar (`:93-493`).
- `x/iter/st/apply_matrix.go:10-130` — row loop assuming **inner stride 1**.
- `x/iter/st/comparison.go:10-474` — strided via `AdvanceOffsets3` (handles any strides).
- `x/iter/st/iterators.go:16-609` — index iterators; `ElementsWindow` `:191-236`, `ElementsWindows` `:253-309`.
- `x/iter/mt/*.go` — chunk the range with `workers.ParallelChunks/Rows` and call `st` per chunk; fall back to `st` when `!workers.ShouldParallelize(n)`.

**Inputs → outputs:** `dst, a, b []T`, `shape []int`, element strides per operand (nil ⇒ row-major contiguous), `op func(T,…) T`. `T` ∈ `types.Number` (float32/64, int*, complex64); comparisons need `NumberReal`. Iterators yield a reused `[]int` index slice (caller must copy).

## Completeness review

- Complete: **no**.
  - Rank-2 strided apply ignores `strides[1]` (inner stride): `x/iter/st/apply_tensor.go:56` passes only `strides[0]` as leading dimension; `x/iter/st/apply_matrix.go:21-25` slices `[i*ld : i*ld+cols]`. Rank ≥3 recurses down to the same rank-2 path. Any transposed/permuted view (`strides=[1,n]`) is computed wrong.
  - `use_mt` cannot build: `x/iter/mt` imports `x/iter/st` → `x/types` → `x/iter/mt` (cycle via `x/types/cast_mt.go`); `x/workers/execute_mt.go:6-29` references undefined `mt`. With default tags `mt` is unreachable (and `workers.ShouldParallelize` is hard-wired `false`, `x/workers/execute_st.go`).
  - `ElementsWindows` returns `func(func([]int, []int, bool) bool)` — **cannot be used with `range`** (Go allows at most 2 yield params); the doc-comment usage `for outIdx, inIdx, isValid := range ElementsWindows(...)` (`st/iterators.go:252`) does not compile (probe compile error). Callable only as a plain function.
- Tested: 57 tests pass, table-driven with expected values (`TestElemApply*`, `TestElemMatApply*`, `TestElemVecApply*`, comparisons, `TestElements*` incl. early-exit and equivalence tests). Strided apply tests use `nil` strides or row-padded `ld` only (e.g. `apply_test.go:22-41`), never a non-unit inner stride, so the defect is not caught.
- Correct: **no** — probe (`tmp/easyrobot/tensor/InfraNotes/iterprobe`):
  - `ElemApplyUnaryStrided` identity, shape [2,2], src strides [1,2] (transpose of `[1 2;3 4]`): got `[1 2 2 3]`, want `[1 3 2 4]`.
  - rank-3 reversed strides: got `[0 1 2 3 1 2 3 4]`, want `[0 4 2 6 1 5 3 7]`.
  - `ElemGreaterThanStrided` with the same transposed strides: correct `[0 1 0 1]` (uses `AdvanceOffsets3`).
  - `ElementsWindows([2,2],[3,3],[4,4],[1,1],[0,0])` yields 36 ✔.

**Defects**
- `x/iter/st/apply_tensor.go:53-57` (and unary `:106-167`, ternary `:183-260`, scalar variants) + `x/iter/st/apply_matrix.go:21-25` — inner stride ignored → wrong results for non-unit inner strides.
- `x/iter/st/iterators.go:217` — `ElementsWindow` allocates `absIndices` per element (probe: 19 allocs for a 3×3 window) despite "slice reused" doc; `:293` allocates `windowOffset` per output position.
- `x/iter/st/iterators.go:252-256` — 3-value yield not rangeable.
- `x/iter/ANALYSIS.md:13-30` (declared) notes `comparison_mt.go` constraint mismatch (`Numeric` vs `NumberReal`) and `apply_tensor_mt.go` uses `Numeric`, which does not exist in `x/types` (only `Number`) → mt files would not compile even without the cycle.

**Embedded suitability:** generics + `op` function value per element (indirect call, not inlined — BCE_ANALYSIS measured ~1.9 ns/elem non-inlined vs 0.5 inlined, declared, see [[bce-loop-experiments]]); stride scratch is stack `[MAX_DIMS]int`; recursion for rank ≥3 slices sub-arrays (no alloc). `ElementsIndices` panics on odd args/rank>16 (`st/iterators.go:326-331`). No goroutines in default build.

## TODOs

- [ ] Honour inner stride in rank-2/≥3 strided apply (fall back to `AdvanceOffsets*` when `strides[last]!=1`) and add transposed-view tests — x/iter/st/apply_tensor.go:53, x/iter/st/apply_matrix.go:21 #p1 #ER-020
- [ ] Fix or remove `use_mt` (cycle x/iter/mt→x/types→x/iter/mt, `Numeric` type) — x/iter/apply_tensor_mt.go:5, x/types/cast_mt.go:7 #p1 #ER-026
- [ ] Make `ElementsWindow(s)` allocation-free (reuse buffers) and expose a 2-value range form — x/iter/st/iterators.go:217,293 #p2
- [ ] Add per-op specialised (non-closure) paths for hot ops (add/mul/relu) to avoid indirect calls — proposal, W2 #p3

## Links

- Module: [[iter-workers]]
- Related: [[worker-pool]], [[shape-strides-iteration]], [[bce-loop-experiments]], [[numeric-cast-and-clamp]]
