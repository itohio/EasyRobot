---
type: algorithm
status: tested
tags: [easyrobot, mat, nnls, ldp, optimisation]
workstream: W9
module: "[[mat-vec]]"
summary: "exact solutions by probe; destroys caller's A"
source: [x/math/mat/nnls.go]
tests: [x/math/mat/nnls_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09): TestNNLS (3 subtests) and TestLDP (3) pass — assert X≥0, small residual, G·X≥H; probe InfraNotes/matprobe2: A=(1 0; 0 1; 1 1), B=(1 2 3) → X=(1 2), rnorm 2e-7; A=I, B=(1 −1) → X=(1 0), rnorm 1 ✔"
---
# Non-negative least squares (NNLS) and least-distance programming (LDP)

**What it does:** Lawson–Hanson active-set NNLS `min ‖AX − B‖ s.t. X ≥ 0`, and LDP `min ‖X‖ s.t. G·X ≥ H` (via NNLS on the dual), wrapping `fp32.Gnnls`/`Gldp`.

**Where:** `x/math/mat/nnls.go:34-108` `NNLS(A, B, dst *NNLSResult, rangeVal)`, `:110-207` `LDP(G, H, dst *LDPResult, rangeVal)`; results `NNLSResult{X, W, RNorm}`, `LDPResult`. Errors `ErrNNLSBadDimensions`, `ErrNNLSMaxIterations` (3n iterations).

**Inputs → outputs:** A m×n, B m (B copied; **A is modified in place**, doc `:27`); X n, dual W n, residual norm. `rangeVal` tolerance parameter passed to the kernel.

## Completeness review

- Complete: yes.
- Tested: 3+3 subtests check feasibility (X ≥ 0, G·X ≥ H) and residual thresholds, not exact X; probe confirms exact solutions on two small cases.
- Correct: yes within tested scope.

**Notes:** caller's A is destroyed (contains Q·A) — easy to misuse; allocates X, W, B copy per call. No non-test caller of `mat.NNLS`/`mat.LDP` in the repo (grep 2026-11-09).

**Embedded suitability:** float32, bounded iterations, a few allocations per call, no goroutines.

## TODOs

- [ ] Copy A or document destructive behaviour at call sites — x/math/mat/nnls.go:27,53 #p3
- [ ] Exact-solution test cases (active constraint) — x/math/mat/nnls_test.go #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-qr]], [[primitive-kernels]]
