---
type: workstream
status: proposed
tags: [easyrobot, workstream, optimisation, tensor, asm]
workstream: W2
---
# W2 — Code-optimisation skills (Go/TinyGo hot paths, tensor kernels, Xtensa/RISC-V asm)

**Goal:** Obsidian-style docs and Claude skills for optimising EasyRobot code, with optimised code running on the ESP32 (Xtensa).

This note records what optimisation work already exists in the repo, what was measured on 2026-11-09, and what a code-optimisation skill should capture. Host for all numbers: Apple M5, darwin/arm64, go1.27.2, `GOTOOLCHAIN=local`.

## What exists

| Artefact | Where | What it is | State |
|---|---|---|---|
| BCE experiments | `x/iter/loop_experiments/bce_experiments.go`, `BCE_ANALYSIS.md` (944 lines) | element-wise 2-D loops in 27 index/reslice/unroll variants, plus amd64 asm (`asm_amd64.s`, `asm_op_amd64.s`) | implemented, benchmarks only; under Rosetta amd64 all 27 variants match the baseline (InfraNotes probe). **Does not build on arm64/riscv64/arm/386/wasm** (asm declared in untagged files, `bce_experiments.go:317,318,374,375`) → [[bce-loop-experiments]] |
| MatMul loop experiments | `loop_experiments/matmul.go` (12 variants + `asmMatMul` `matmul.go:5`), `asm_matmul_amd64.s`, `MATMUL_ANALYSIS.md` | 100×100 matmul variants; winner "Exp7: unroll middle (k) loop by 4" (ikj order, contiguous B rows) | partial: asm exact, but `MatMulOptimizedTranspose` is wrong whenever k ≠ n (`matmul.go:57-62`) — no correctness tests, so a benchmarked variant was wrong; amd64 only → [[matmul-asm-experiment]] |
| SIMD convert | `x/types/st/convert_asm_amd64.s`, `CONVERT_PLAN.md`, `dispatch_overhead_analysis.md`, `dispatch_optimization_issue.md` | amd64 asm "int16→float32"; analysis of generic type-switch dispatch overhead | **stub**: the asm writes the constant 3.05e-5 into every output element (checked under Rosetta); declared only in a test file, which breaks `x/types/st` tests on arm64 (`convert_bench_test.go:583`) → [[simd-convert-amd64]], [[numeric-cast-and-clamp]] |
| Iterator/threading analysis | `x/iter/ANALYSIS.md`; `use_mt` build tag (`x/iter/iterators_{st,mt}.go`, `x/workers/execute_{st,mt}.go`) | single- vs multi-thread iteration and chunking | default st builds; **`use_mt` does not build** (import cycle, undefined types); rank ≥ 2 strided apply drops the inner stride → [[tensor-iterators]], [[worker-pool]] |
| FP32 BLAS-style primitives | `x/math/primitive/fp32/level{1,2,3}.go`, `BLAS.md`, `LA.md`, `OPS.md` | Gemm_NN/NT/TN/TT, Syrk, Trmm, Gemv…; inner loops unrolled ×4 | pure Go, no asm; Gemm correct (tests + probes), `Trmv`/`Trmm` wrong → [[fp32-gemm]], [[fp32-blas-level1-2]] |
| Benchmark reports | `primitive/fp32/{BENCHMARK_REPORT,PERFORMANCE_REPORT}.md`, `tensor/eager/BENCHMARK_REPORT.md`, `tensor/BENCHMARKS.md`, `nn/layers/{BENCHMARK_REPORT,OPTIMIZATION_ANALYSIS,LEARNING_OPTIMIZATION_PLAN}.md` | declared results | `PERFORMANCE_REPORT.md` is **stale**: compares an "assembly-accelerated" default path with a `use_go` tag and cites `level3_go.go`, `conv_go.go`, `pkg/core/...` — none exist at HEAD; fp32 has no `.s` files |
| Benchmarks in code | e.g. `x/math/tensor/bench_test.go`, `eager/bench_test.go` (2267 lines), `nn/layers/conv_benchmark_test.go`, `x/iter/*_bench_test.go`, `x/types/*_bench_test.go`, `shape/*_bench_test.go` | Go benchmarks | runnable |
| Fixed-point primitives | `x/math/primitive/{qi,qi8,qi16,qi32,qi64,qu8,bytes}` | package declarations only (7 lines each) | **stub** → [[quantised-dtypes-and-fixed-point]] |

### Claims in the analyses (declared, amd64, not re-run here)

- `BCE_ANALYSIS.md`: element-wise contiguous 2-D best = flatten to 1-D + reslice (33–38 % faster than baseline); strided best = row slices + reslice + range + unroll (37 %); inlined amd64 asm 57 % faster; "pre-compute offsets without range" 36 % slower. Matmul section: asm 473 µs (80 % faster than naive), optimised Go 973 µs.
- `MATMUL_ANALYSIS.md`: Exp7 (ikj, unroll k by 4) 625 650 ns/op, "59 % faster than assembly", 74 % faster than naive; 4×4/8×8 blocking 76–90 % *slower* at 100×100. This contradicts `BCE_ANALYSIS.md`'s "asm is 80 % faster" (0.47 ms vs 1.54 ms asm timings); neither records the host CPU; both at 100×100 only, where all operands fit in L1/L2, so neither result says anything about blocking for large matrices.
- Other recommendations in `BCE_ANALYSIS.md`: recursive decomposition for rank ≥ 3, stack scratch instead of `make`, avoid a closure call per element (1.92 vs 0.51 ns/element) — the closure-per-element style is exactly what `x/iter` uses today.
- Neither analysis tested correctness of its variants (see the `MatMulOptimizedTranspose` bug above).

## Measured on 2026-11-09

The host is shared (load average 9–30 during these runs, other agents testing), so absolute numbers vary up to ~3× between runs; ratios measured in the same run are the reliable part.

1. **Eager MatMul (production path `eager.MatMul` → `fp32.Gemm_NN`) vs Gorgonia graph (gonum BLAS)**, `go test -count=1 -run '^$' -bench 'BenchmarkMatMul_(Eager|GorgoniaGraph)_(Medium|Large)' -benchtime 2s -benchmem ./x/math/tensor/` (01:45, load ≈ 15):
   eager 128×128 1.56 ms (2.7 GFLOP/s, 6 allocs/65 KB); 512×512 209 ms (**1.3 GFLOP/s**, 6 allocs/1 MB).
   Gorgonia 128×128 0.52 ms (8.0 GFLOP/s, 50 allocs); 512×512 15.8 ms (**17 GFLOP/s**, 173 allocs) — 13× faster than eager at 512 despite graph overhead.
   An earlier, more contended run (`-benchtime 5x`, 01:30) gave eager 32×32 40 µs, 128 1.70 ms, 512 416 ms; Gorgonia 512 42 ms — same ordering.
2. **Loop-order probe** (verbatim copies of `Gemm_NN`'s loop `level3.go:38-65` and `MatMulExp7` `matmul.go:278-321`; `TestSame` checks equal results for n = 5, 33, 128 → pass). Calibration: in the same session the copied ijk loop and the repo benchmark agree (32×32: 11.5 µs vs `BenchmarkMatMul_Eager_Small_32x32` 11.7 µs; 128×128: 1.187 ms vs 1.194 ms). `go test -bench . -benchtime 300ms` (01:45):

   | n | Gemm_NN ijk | Exp7 ikj | speed-up |
   |---|---|---|---|
   | 32 | 11.5 µs (5.7 GFLOP/s) | 5.3 µs (12.3) | 2.2× |
   | 128 | 1.19 ms (3.5) | 0.30 ms (13.8) | 3.9× |
   | 512 | 112.6 ms (2.4) | 22.1 ms (12.2) | 5.1× |

   A first, heavily contended run gave 3.0×/3.4×/6.2×. Cause: `Gemm_NN` walks B down a column (`b[pb+ldB]`, stride N floats), so every k step touches a new cache line once rows exceed L1; Exp7 streams B rows contiguously, keeps 4 A values in registers and its inner loop is a contiguous axpy the compiler handles well. Plain-Go Exp7 gets within ~1.5× of gonum's assembly at 512.
   → The repo's own winning experiment was never ported into `fp32.Gemm_NN`, which feeds MatMul, Conv2D (im2col + GEMM) and Dense.
3. **Portability:** `CGO_ENABLED=0 go build` of tensor/types, tensor/eager, tensor, primitive/..., nn/..., mat, vec, iter/..., workers/..., types/... for linux/riscv64, linux/arm, linux/386 fails **only** in `x/iter/loop_experiments` (missing asm bodies); the numeric core compiles for riscv64 (K230 Linux core) and 32-bit arm. js/wasm additionally fails in `x/types/st` because int-size files are gated by explicit GOARCH lists (`st/convert_i32.go:1` etc.; `cast.go:82: undefined: elemConvertToInt`) — see [[W9-go-nn-framework]].

## Gaps

- No blocked/packed GEMM, no GEMV specialisation (decode-time LLM work is GEMV), no int8 kernels.
- Asm without `_generic.go`/`!amd64` fallbacks breaks non-amd64 builds; asm lives only in experiments, not in production kernels.
- No benchmark harness that records GFLOP/s or compares against a reference BLAS; reports are hand-written and drift from code.
- No allocation budget per op in tests (`testing.AllocsPerRun`), although TinyGo needs zero-alloc hot paths.
- Nothing measured on Xtensa or RISC-V.

## What a code-optimisation skill should capture

1. **Measure first, on the target.** `go test -run '^$' -bench X -benchmem -count 5` + `benchstat`; report GFLOP/s or ns/element, not only ns/op; record CPU, Go version, GOARCH, tags; no concurrent load (this run shows how contention distorts numbers).
2. **Loop order and memory access before asm.** ikj / contiguous inner loop, register-blocked micro-kernel (e.g. 4×4 or 8×4 accumulators), cache blocking only for operands larger than L2 (and prove it at sizes ≥ 256, not 100).
3. **BCE idioms** (from `BCE_ANALYSIS.md`): reslice to exact length (`s = s[:n]`), row slices, `range` over the slice being written, hoist `_ = s[n-1]`; check with `go build -gcflags=-d=ssa/check_bce`.
4. **Allocation discipline.** `dst`-style APIs, caller-provided workspaces for im2col/temporaries, `testing.AllocsPerRun == 0` assertions; no `fmt.Sprintf` in hot paths.
5. **Type dispatch cost.** Hoist `any` type switches out of loops; specialise float32 paths; see `x/types/st/dispatch_overhead_analysis.md`.
6. **Assembly rules.** Every `.s` needs a pure-Go fallback with the opposite build tag and a test comparing both; keep asm behind one small kernel (GEMM micro-kernel, dot, conversion); Go asm cannot target Xtensa (gc has no Xtensa port), so ESP32 SIMD means TinyGo + inline asm/intrinsics or a C library — a research question, see [[W9-go-nn-framework]].
7. **TinyGo specifics.** float32 only on ESP32 FPU; avoid reflection-heavy packages (testify leaks into `nn/layers` today); prefer fixed-size arrays and stack allocation; check binary size.
8. **Correctness gate.** Each optimised kernel keeps the naive reference and a randomised equality test (tolerance stated).

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Add GFLOP/s metric + `AllocsPerRun` checks to the matmul/conv benchmarks — x/math/tensor/bench_test.go #p3
- [ ] GEMV fast path for M=1 (LLM decode, Dense with batch 1) — fp32/level2.go, eager/linalg.go #p2
- [ ] Rerun the gemmprobe on an idle machine and on the ESP32-S3/K230 once available #p3

## Inferred objectives (proposals)

- **Proposal:** a `go-kernel-optimisation` skill with the checklist above plus scripts: bench → benchstat → BCE report → alloc check, and a "fallback required" lint for `.s` files.
- **Proposal:** a reference-kernel test pattern (naive vs optimised, random shapes incl. odd sizes and strided views) applied first to GEMM, Conv2D, Softmax, RMSNorm.
- **Proposal:** treat `fp32.Gemm_*` as the single hot spot to optimise (MatMul, Conv2D, Conv2DTransposed, Dense all reduce to it) before any per-op tuning.

## Research questions for ideation

1. TinyGo on Xtensa LX7: can inline asm reach the ESP32-S3 PIE vector instructions, and what speed-up does a 4×4 int8 GEMM micro-kernel give over scalar TinyGo?
2. For RISC-V (K230 C908 with RVV 1.0): gc does not autovectorise; is Go asm for RVV supported by the gc assembler in Go 1.25+, or is a C shim needed?
3. What is the best pure-Go GEMM achievable on arm64 (register blocking without asm) vs gonum's asm, and does it carry over to TinyGo/LLVM?
4. Should conversions (`x/types/st`) keep asm at all, given the dispatch-overhead analysis says type-switch overhead dominates small slices?
