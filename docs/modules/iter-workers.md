---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, iter, workers, concurrency, optimisation]
workstream: [W2, W9]
packages: [x/iter, x/iter/st, x/iter/mt, x/iter/loop_experiments, x/workers, x/workers/mt]
test_result: pass
---
# iter-workers

**Purpose:** generic element-wise iteration over flat and strided buffers (`ElemApply*`, comparisons, fills, window iterators) used by the FP32 primitives and type conversions, with an optional multi-threaded variant (`-tags use_mt`) backed by a worker pool; plus loop/asm micro-benchmark experiments.

**Packages (non-test LOC):**

| Package | Files | LOC | Notes |
|---|---|---|---|
| `x/iter` | 15 | 1592 | public generic API; `iterators_st.go`/`iterators_mt.go`, `apply_*_{st,mt}.go`, `comparison_{st,mt}.go` select st or mt by `use_mt` |
| `x/iter/st` | 5 | 2075 | single-threaded implementations (flat, matrix, rank-N strided via `shape.AdvanceOffsets*`) |
| `x/iter/mt` | 5 | 1635 | chunked multi-threaded implementations |
| `x/iter/loop_experiments` | 2 + 3 `.s` | 894 + 469 asm | BCE and matmul experiments, `BCE_ANALYSIS.md`, `MATMUL_ANALYSIS.md`; amd64 asm only |
| `x/workers` | 3 | 559 | `WorkerPool`, `Execute`, `ParallelChunks/Rows/TensorChunks`, `ShouldParallelize`; `execute_{st,mt}.go` |
| `x/workers/mt` | 1 | 83 | global pool, `minParallelSize = 128·NumCPU` |

Declared analysis: `x/iter/ANALYSIS.md`.

## Public API

- `iter.ElemApplyUnary/Binary/Ternary[T Number](dst, a, (b,) n, op func(...) T)`, `…Scalar`, `…Strided(dst, src, shape, stridesDst, stridesSrc, op)`, `ElemEqual/Greater…`, `ElemFill(Strided)`, window/`Elements*` iterators (Go 1.23 iterator funcs).
- `workers.Execute(total, fn(start,end))`, `ParallelChunks`, `ParallelRows`, `ParallelTensorChunks`; `WorkerPool` with `WithWorkers`, `WithChunkSizer`, `WithTargetChunkSize`.
- Composition: `primitive/fp32` element-wise kernels → `x/iter` (≈24 call sites) → `x/iter/st` (default) or `x/iter/mt` → `x/workers`.

## Algorithms

<!-- vault:algorithms h=15783f4180f9 -->

Counts: 0 tested · 1 implemented · 3 partial · 0 stub · 0 unverified (4 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[bce-loop-experiments]] | implemented | build-fail | benchmarks only; 27 variants match baseline under Rosetta amd64; asm declared in untagged files |
| [[matmul-asm-experiment]] | partial | build-fail | asm matmul exact; `MatMulOptimizedTranspose` wrong when k ≠ n (`matmul.go:57-62`); Exp7 loop order… |
| [[tensor-iterators]] | partial | pass | rank ≥ 2 strided apply ignores the inner stride (`x/iter/st/apply_tensor.go:53-57`… |
| [[worker-pool]] | partial | pass | nested `Execute` from a callback deadlocks (`worker_pool.go:411`); default-build… |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Default build: no goroutines, no `sync` in the hot path, generics + closures per element (indirect call per element; `BCE_ANALYSIS.md` measures 1.92 vs 0.51 ns/element for closure vs inlined).
- **`-tags use_mt` is broken** across the tree: import cycle `x/iter/mt → x/types → x/iter/mt`, undefined `Numeric` in `x/iter/*_mt.go`, `x/workers/execute_mt.go:6` bad import (details in [[tensor-iterators]], [[worker-pool]]). Nothing multi-threaded is usable today.
- `x/iter/loop_experiments` fails to build on every non-amd64 GOARCH (arm64, riscv64, arm, 386, wasm: `bce_experiments.go:317,318,374,375`, `matmul.go:5` missing function bodies).
- TinyGo: the st path is suitable in principle (generics are supported by TinyGo); closure-per-element is slow on a 240 MHz core; workers/mt are irrelevant on ESP32.

## Tests and evidence

- `go test -count=1 -json ./x/iter/... ./x/workers/...` (2026-11-09): `x/iter` 57 pass; `x/workers` 19 pass; `x/iter/st`, `x/iter/mt`, `x/workers/mt` no test files; `x/iter/loop_experiments` build failed (darwin/arm64).
- InfraNotes ran `loop_experiments` under `GOARCH=amd64` (Rosetta): builds, no tests; probe sources `~/src/tmp/easyrobot/tensor/InfraNotes/{loopexp,matmulprobe,iterprobe}`.
- `go build -tags use_mt` not attempted by the lead; breakage is from source reading + InfraNotes notes.

## API notes (ease of use, composability)

- Good: one generic API for all numeric types; strided variants take explicit shape/strides, so they compose with views; `dst`-first style matches the tensor API.
- Weak: closure-per-element API blocks inlining and SIMD; offsets are not part of the strided API (callers must pre-slice, and eager often forgets, see [[tensor-views-strides]]); st/mt switching by build tag means a whole program is single- or multi-threaded, and the mt side is unmaintained; `Elements*` window iterators allocate.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Non-closure specialised paths for hot ops (add/mul/relu/axpy) — x/iter/st #p3

## Inferred objectives (proposals)

- **Proposal:** drop build-tag threading in favour of an explicit `Executor` passed to kernels (nil = inline), which also works on TinyGo and K230.
- **Proposal:** keep `loop_experiments` as the W2 benchmark lab, but with correctness tests and per-arch results tables (amd64, arm64, riscv64/K230, TinyGo/ESP32).
