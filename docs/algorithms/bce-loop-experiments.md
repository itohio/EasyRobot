---
type: algorithm
status: implemented
tags: [easyrobot, iter, performance, bce, asm]
workstream: [W2, W9]
module: "[[iter-workers]]"
summary: "benchmarks only; 27 variants match baseline under Rosetta amd64; asm declared in untagged files"
source: [x/iter/loop_experiments/bce_experiments.go, x/iter/loop_experiments/asm_amd64.s, x/iter/loop_experiments/asm_op_amd64.s, x/iter/loop_experiments/BCE_ANALYSIS.md]
tests: [x/iter/loop_experiments/bce_experiments_test.go]
test_result: build-fail
complete: yes
correct: unknown
evidence: "go test -count=1 ./x/... (2026-11-09): build-fail darwin/arm64 (bce_experiments.go:317,318,374,375 missing function body); GOARCH=amd64 go test (Rosetta) → ok, [no tests to run]; probe InfraNotes/loopexp (amd64/Rosetta): 20 contiguous + 7 strided variants match baseline"
---
# Bounds-check-elimination loop experiments

**What it does:** Benchmark-only package comparing ~30 loop shapes for a 2D element-wise op `dst = src*src + 1` (1000×1000 float32): nested loops, reslicing hints (`_ = s[n-1]`), row slices, `for range`, flattening, precomputed offsets, 4× unrolling, and amd64 assembly (with Go-callback `op` and with the op inlined in SSE).

**Where:** `x/iter/loop_experiments/bce_experiments.go:7-409` (`BaselineNestedLoops` `:18`, `BCE_*` variants, `Strided_*` variants with `ldDst/ldSrc`, `BCE_Assembly*` wrappers); asm decls without body `:317-318` (`asmOpLoop`, `asmOpLoopUnrolled`), `:374-375` (`asmOpDirect`, `asmOpUnrolled`); asm `asm_amd64.s:8,63`, `asm_op_amd64.s:12,50`. Benchmarks `bce_experiments_test.go:30-360` (36 `Benchmark_*`, no `Test*`).

**Inputs → outputs:** `(dst, src []float32, rows, cols int[, ldDst, ldSrc int])` → dst written in place.

## Completeness review

- Complete: yes for its stated scope (experiment harness).
- Tested: **no tests**, benchmarks only. Package fails to compile on any non-amd64 GOARCH because the asm functions are declared in untagged `.go` files (`bce_experiments.go:317,318,374,375`, `matmul.go:5`); the `.s` files are amd64-only (`// +build amd64`). This is the darwin/arm64 build failure in the 2026-11-09 run.
- Correct: unknown→**partially checked**: probe under `GOARCH=amd64` (Rosetta) compared all 20 contiguous variants (incl. 6 assembly) and 7 strided variants against `BaselineNestedLoops`/`StridedBaseline` on a 7×13 (ld 16) case: 0 mismatches. `asmOpLoop` calls a Go func value from ABI0 asm via `MOVQ (DX), AX; CALL AX` in a `NOSPLIT $32` frame (`asm_amd64.s:8-61`) without the ABIInternal register setup (R14=g, X15=0) [INFERENCE: ABI-unsafe, works by accident].

**Declared findings (BCE_ANALYSIS.md, claims only; host CPU not recorded, amd64, Nov 2025):**
- Contiguous: SSE asm with inlined op fastest (0.94 ms vs 3.26 ms baseline, −57 %, `BCE_ANALYSIS.md:16`); best Go = flat 1D loop + reslice (2.06 ms, −38 %, `:18`); row-slices alone are *slower* than baseline (+15 %, `:37`).
- Strided: row slices + reslice + range + 4× unroll best (−37 %, `:44`); precomputed offsets worst (+18 %, `:51`).
- Asm that calls the Go `op` per element gains only ~25 % (`:29-32`) — call overhead dominates (1.92 vs 0.51 ns/op, `:57-58`).
- Claims asm beats the "theoretical minimum" by 47 % (`:497`) — the minimum was derived from scalar read/write benchmarks, so this mostly shows the minimum is wrong.
- Recommends flatten-when-contiguous, row-slices+unroll for strided, recursive decomposition for rank ≥3 (`:484-491`), stack scratch instead of `make` (`:509-564`). The production iterators follow the decomposition idea ([[tensor-iterators]]) but still call `op` per element.
- The file also contains an unrelated Conv2D investigation (im2col+GEMM → direct conv, "30 s → 3.8 ms", `:735-850`).

**Embedded suitability:** findings are amd64-specific; Xtensa LX6/LX7 has no bounds-check hardware cost model comparable to x86 and TinyGo's LLVM backend does its own BCE — re-measure on target before adopting. Unrolling increases flash.

## TODOs

- [ ] Move asm declarations to `*_amd64.go` and add pure-Go fallbacks (`*_other.go`) so the package builds on arm64/riscv64/xtensa — x/iter/loop_experiments/bce_experiments.go:317 #p1 #ER-026
- [ ] Add correctness tests (variant == baseline) so benchmarks can't silently measure wrong code — x/iter/loop_experiments/bce_experiments_test.go #p2
- [ ] Re-run on darwin/arm64, K230 and TinyGo/ESP32 and record host — BCE_ANALYSIS.md:10 #p2
- [ ] Drop/replace callback-calling asm (ABI0→ABIInternal call) — x/iter/loop_experiments/asm_amd64.s:8 #p3

## Links

- Module: [[iter-workers]]
- Related: [[matmul-asm-experiment]], [[tensor-iterators]], [[simd-convert-amd64]]
