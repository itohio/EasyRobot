---
type: algorithm
status: partial
tags: [easyrobot, iter, performance, matmul, asm, simd]
workstream: [W2, W9]
module: "[[iter-workers]]"
summary: "asm matmul exact; `MatMulOptimizedTranspose` wrong when k ≠ n (`matmul.go:57-62`); Exp7 loop order…"
source: [x/iter/loop_experiments/matmul.go, x/iter/loop_experiments/asm_matmul_amd64.s, x/iter/loop_experiments/MATMUL_ANALYSIS.md]
tests: [x/iter/loop_experiments/matmul_test.go]
test_result: build-fail
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): build-fail darwin/arm64 (matmul.go:5 missing function body); probes InfraNotes/matmulprobe + loopexp (GOARCH=amd64 via Rosetta, 2026-11-09): asmMatMul exact on 4 shapes; MatMulOptimizedTranspose wrong for k≠n"
---
# Matmul loop/asm experiment

**What it does:** Benchmarks 17 float32 GEMM variants `C[m×n] = A[m×k]·B[k×n]` (row-major): naive triple loop, row-slice/BCE "Optimized", B-transpose, flattened, Exp1–Exp12 (row bases, unrolling of j or l by 4, 4×4/8×8 blocking, column-wise B), and an amd64 SSE kernel.

**Where:** `x/iter/loop_experiments/matmul.go:5` (`asmMatMul` decl, no body, untagged), `:9` `MatMulNaive`, `:23` `MatMulOptimized`, `:52-83` `MatMulOptimizedTranspose`, `:86-110` `MatMulFlattened`, Exp1–Exp12 below; `asm_matmul_amd64.s:9-168` (`asmMatMul`: zero C row, for each l broadcast `A[i,l]` (`SHUFPS $0`), `MULPS/ADDPS` 4 columns of B/C per step with `MOVUPS`, scalar remainder). Benchmarks `matmul_test.go:25-144` (100×100).

**Inputs → outputs:** `(C, A, B []float32, m, k, n int)`; asm takes `*float32` pointers; C overwritten.

## Completeness review

- Complete: yes (experiment scope). No production caller; `x/math/primitive/fp32` GEMM is separate ([[primitive-kernels]]).
- Tested: benchmarks only, no `Test*`. Arm64 build break as in [[bce-loop-experiments]].
- Correct: **no** (one variant):
  - `asmMatMul` vs naive: max |err| = 0 on [3,5,7], [4,4,4], [2,3,9], [16,16,16] (probe, amd64 under Rosetta).
  - All Go variants match naive on [5,7,9], [8,8,8], [3,1,17] **except `MatMulOptimizedTranspose`**: max err 0.648 on [5,7,9]. Cause: transpose loop writes `BT[i*n+j] = B[j*k+i]` (`matmul.go:57-62`) — indices swapped; correct only when k == n. The benchmarks (100×100, square) hide it.
  - It also allocates `k*n` floats per call (`matmul.go:54`).

**Declared findings (MATMUL_ANALYSIS.md, claims only, amd64, host not recorded):**
- Exp7 (unroll middle `l` loop by 4) fastest: 0.63 ms vs 2.43 ms naive (−74 %) and −59 % vs the asm kernel (`MATMUL_ANALYSIS.md:11`).
- SSE asm 1.54 ms (−36 % vs naive, `:19`) — but `BCE_ANALYSIS.md:392` reports the same asm at 0.47 ms (−80 %): the two documents contradict each other.
- Blocking 4×4/8×8 slower than naive at 100×100 (`:21,23`); flattened indices worst (+185 %, `:26`).
- "Why Exp7 beats assembly" (`:226-243`): asm reloads/stores C each l iteration and recomputes indices with IMUL; Go keeps 4 A values in registers.

**Embedded suitability:** SSE asm is amd64-only. The Exp7 idea (register-blocked l-unroll, no allocation) is portable Go and applicable to ESP32/K230 GEMM; needs re-measurement with TinyGo/LLVM.

## TODOs

- [ ] Fix `MatMulOptimizedTranspose` indexing (`BT[j*k+i] = B[i*n+j]`) or delete — x/iter/loop_experiments/matmul.go:57-62 #p1 #ER-020
- [ ] Tag `asmMatMul` decl amd64-only + Go fallback — x/iter/loop_experiments/matmul.go:5 #p1 #ER-026
- [ ] Add value tests with non-square shapes — x/iter/loop_experiments/matmul_test.go #p2
- [ ] Reconcile contradictory asm timings (BCE_ANALYSIS.md:392 vs MATMUL_ANALYSIS.md:19), port Exp7 to fp32 GEMM if confirmed on arm64 — W2 proposal #p2

## Links

- Module: [[iter-workers]]
- Related: [[bce-loop-experiments]], [[simd-convert-amd64]], [[primitive-kernels]]
