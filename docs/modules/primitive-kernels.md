---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, primitive, blas, kernels, fp32, quantisation]
workstream: [W9, W2]
packages: [x/math/primitive/fp32, x/math/primitive/c64, x/math/primitive/qi, x/math/primitive/qi8, x/math/primitive/qi16, x/math/primitive/qi32, x/math/primitive/qi64, x/math/primitive/qu8, x/math/primitive/bytes]
test_result: pass
---
# primitive-kernels

**Purpose:** slice-level numeric kernels with BLAS-like signatures (`dst, src, strides/ld, sizes`) that the eager [[tensor]] backend calls: FP32 BLAS levels 1–3, LAPACK-style decompositions, convolutions/pooling/im2col, activations, normalisations, element-wise/broadcast/reduction helpers, FFT and 1-D/2-D DSP convolution/correlation; complex64 FFT/convolution; placeholders for fixed-point/quantised types.

**Packages (non-test LOC):**

| Package | Files | LOC | Content |
|---|---|---|---|
| `x/math/primitive/fp32` | 18 | 7694 | `level1.go`, `level2.go`, `level3.go` (Gemm_NN/NT/TN/TT, Syrk, Trmm), `batched.go`, `la.go` (1452: LU/Getri, Geqrf/Orgqr, Gesvd, Gepseu, Gnnls), `tensor.go` (2038: Conv1D/2D, transposed, kernel grads, pooling, im2col), `tensor_elementwise/broadcast/reduction.go`, `activations.go`, `normalization.go` (774), `dsp_fft.go`, `dsp_convolution.go`, `dsp_correlation.go`, `array.go`, `vector.go`; `Pool = buffer.Pool[float32]` |
| `x/math/primitive/c64` | 5 | 767 | complex64 FFT (+LUT), Convolve1D/2D, CrossCorrelate1D/2D |
| `x/math/primitive/qi8` | 7 + 3 tests | 651 | int8 NHWC kernels since ER-NN-KERNELS N1: Dot, FullyConnected, Conv2D, DepthwiseConv2D, Max/AvgPool2D, ReLU/ReLU6, exact per-channel requantisation ([[int8-nn-kernels]]); plus the old `Pool` |
| `qi`, `qi16`, `qi32`, `qi64`, `qu8`, `bytes` | 1 each | 7 each | only `var Pool = buffer.Pool[T]{}`; `bytes/bytes.go` declares `package qi8` |

No assembly, no build tags. Declared docs: `primitive/SPEC.md` (811), `primitive/PLAN.md`, `fp32/{BLAS,LA,OPS}.md`, `fp32/BENCHMARK_REPORT.md`, `fp32/PERFORMANCE_REPORT.md` (stale: describes an assembly default path and files that do not exist, see [[W2-code-optimisation]]), `c64/OPS.md`.

## Public API

- BLAS: `Axpy(y, x, strideY, strideX, n, alpha)`, `Dot`, `Nrm2`, `Asum`, `Scal`, `Iamax`; `Gemv_N/T`, `Ger`, `Symv`, `Trmv`; `Gemm_NN/NT/TN/TT(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)`, `GemmBatched/Strided`, `Syrk`, `Trmm`.
- LA: `Getrf/Getri`, `Geqrf/Orgqr`, `Gesvd`, `Gepseu`, `Gnnls`, errors `ErrSingularMatrix`, `ErrUnsupportedOperation`.
- NN kernels: `Conv2D(output, input, weights, batch, inC, outC, …)`, `Conv2DTransposed`, `Conv2DKernelGrad`, `Conv1DKernelGrad`, `Im2Col/Col2Im`, `MaxPool2D(+Backward)`, `AvgPool1D/2D/3D(+Backward)`, `AdaptiveAvg/MaxPool*`, `BatchNormForward/Grad`, `LayerNorm*`, `RMSNorm*`, `InstanceNorm*`, `GroupNorm*`, activation helpers, `Argmax/Argmin`, reductions, `BroadcastStrides(To)`.
- DSP: `FFT/IFFT`, `FFTWithLUT`, `NewFFTLUT`, `Convolve1D/2D(Add)`, `CrossCorrelate1D/2D(Add)` (fp32 and c64).
- Composition: callers pass raw slices + layout; no tensor types. The eager backend must pass offset-adjusted slices and leading dimensions — it mostly does not (see [[tensor-views-strides]]).

## Algorithms

<!-- vault:algorithms h=38bdf327c7bf -->

Counts: 3 tested · 0 implemented · 3 partial · 1 stub · 0 unverified (7 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[fp32-gemm]] | tested | pass | correct; i-j-k loop with column-strided B, no blocking/SIMD (2.3–4.8 GFLOP/s on M5 by the kernel… |
| [[int8-nn-kernels]] | tested | pass | Generic int8 NHWC kernels (dot, FC, conv, depthwise, pools, ReLU/ReLU6/LeakyReLU, Add, Mul, Mean, Requantize, Logistic, Softmax, Concat, Pad, nearest and bilinear resize) with exact single-rounding or TFLite requantisation; bit-exact reference for SIMD paths; allocation-free. |
| [[primitive-fft]] | tested | pass | radix-2, power-of-2; non-LUT twiddle recurrence drifts (1.6e-2 vs 2e-4 at n = 16384) |
| [[fp32-blas-level1-2]] | partial | pass | `Trmv` transpose paths wrong and `TestTrmv` encodes the wrong outputs (`level2.go:211-279`) |
| [[fp32-linear-algebra-decompositions]] | partial | pass | QR does not reconstruct A (error 3.1, `la.go:542-660`); `Gesvd` returns V not Vᵀ, unsorted; pinv… |
| [[primitive-convolution-correlation]] | partial | pass | correlation only non-negative lags; c64 correlation not conjugated… |
| [[quantised-dtypes-and-fixed-point]] | stub | pass | qi*/qu8/bytes are pool declarations only; no scale/zero-point; `CloneTensorDataTo` returns zeros… |

<!-- /vault:algorithms -->

Kernels also documented from the tensor side (they live in `fp32/tensor.go` / `normalization.go` / `activations.go`): [[conv2d-kernel]], [[conv1d-kernel]] (`Conv1DKernelGrad` layout bug `fp32/tensor.go:349-370`), [[conv2d-transposed-kernel]], [[im2col-col2im]], [[pooling-kernels]] (AdaptiveAvgPool `fp32/tensor.go:897-900`), [[activation-kernels]], [[softmax-kernel]], [[batch-norm]], [[layer-norm]], [[rms-norm]], [[instance-and-group-norm]] (all γ-weighted gradient sums wrong, `fp32/normalization.go`).

## Build targets and dependencies

- Pure Go, float32 math via `chewxy/math32` — but several activation/softmax paths call float64 `math.Exp`/`math.Tanh` (`fp32/activations.go:57,83,121,157,196`), soft-float on ESP32.
- No goroutines; deps `x/iter` (st), `x/types`, `x/buffer`, `x/math/shape`. Builds for linux/riscv64/arm/386; wasm fails via `x/types/st`.
- Allocation: BLAS routines allocate nothing; conv/pool/LA paths use `fp32.Pool` or `make` for workspaces (per call); `Gesvd`/QR allocate.
- **Int8 kernels** exist only in `qi8` (NHWC, generic Go, [[int8-nn-kernels]]); no int16 kernels; no SIMD/asm yet (ESP32-S3 PIE is ER-NN-KERNELS N2).

## Tests and evidence

- `go test -count=1 -json ./x/math/primitive/...` (2026-11-09): fp32 97 top-level pass (304 incl. subtests), c64 30 pass (76 incl. subtests); qi*/qu8/bytes no test files.
- Several tests are weak or wrong: `TestTrmv` asserts the implementation's wrong outputs; QR/SVD tests check identity or orthogonality only; correlation tests encode zero negative lags. Probe sources: `~/src/tmp/easyrobot/tensor/kernelnotes/{,qr,p2,p3}`.
- Benchmarks: `fp32/dsp_fft_bench_test.go`, `tensor_bench_test.go`, `c64/dsp_fft_bench_test.go`; GEMM numbers in [[fp32-gemm]] and [[W2-code-optimisation]].

## API notes (ease of use, composability)

- Good: BLAS-shaped signatures are familiar, allocation-free and easy to port to C/asm or ESP-DSP equivalents; leading-dimension parameters allow sub-matrix work without copies.
- Weak: no length validation (bad sizes panic with index errors); argument orders differ between families (`Im2Col` takes pad before stride; `Axpy(y, x, …)` vs `Gemm(c, a, b, …)`); quantised packages exist by name only, which suggests a capability that is not there; correlation/convolution semantics differ from NumPy/SciPy without saying so.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Fix `Trmm` transpose and right-side branches — level3.go:396-463,539-603 #p1 #ER-020
- [x] Implement int8 GEMV/conv with int32 accumulation — x/math/primitive/qi8 ([[int8-nn-kernels]]) #p2
- [ ] Mark PERFORMANCE_REPORT.md historical — x/math/primitive/fp32/PERFORMANCE_REPORT.md:13-16 #p3

## Inferred objectives (proposals)

- **Proposal:** treat this package as the porting surface for ESP32-S3 and K230: a small, fixed kernel set (GEMM, GEMV, Conv2D NHWC, depthwise, pooling, requantise) with a pure-Go reference implementation and per-target fast paths behind build tags, each checked against the reference.
- **Proposal:** add an int8 path modelled on TFLite-micro/ESP-NN semantics (int8 weights, per-channel scales, int32 accumulators, requantisation) so imported quantised models can run natively.
