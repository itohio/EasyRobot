---
type: algorithm
status: tested
tags: [easyrobot, nn, int8, quantisation, kernels, embedded]
workstream: W9
module: "[[primitive-kernels]]"
summary: "Generic int8 NHWC kernels (dot, FC, conv, depthwise, pools, ReLU/ReLU6) with exact single-rounding per-channel requantisation; bit-exact reference for SIMD paths; allocation-free."
source: [x/math/primitive/qi8/quant.go, x/math/primitive/qi8/act.go, x/math/primitive/qi8/shape.go, x/math/primitive/qi8/dot.go, x/math/primitive/qi8/conv.go, x/math/primitive/qi8/pool.go]
tests: [x/math/primitive/qi8/reference_test.go, x/math/primitive/qi8/kernels_test.go, x/math/primitive/qi8/bench_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/primitive/qi8 → ok (2026-11-09): 0 mismatches vs an exact big.Float float-then-quantise reference over 15 310 random cases (≈2.5 M outputs); AllocsPerRun 0 for every kernel; tinygo 0.43 -print-allocs=qi8 empty for xiao-esp32s3 and the UnitV (K210) target (ER-NN-KERNELS N1)"
---
# Int8 NN kernels (generic reference)

**What it does:** the generic int8 inference kernels of the NN layer (ER-NN-KERNELS N1): `Dot`, `FullyConnected`, `Conv2D` (any kernel, stride, same/valid padding), `DepthwiseConv2D` (channel multiplier 1), `MaxPool2D`, `AvgPool2D`, `ReLU`, `ReLU6`, plus per-channel requantisation (`Requant`, `QuantizeMultiplier`) and fused-activation clamp helpers. Pure Go, no allocation, TinyGo-buildable. It is also the bit-exact reference for the ESP32-S3 PIE kernels (N2) and the K210 CPU path.

**Where:** `x/math/primitive/qi8` (`quant.go`, `act.go`, `shape.go`, `dot.go`, `conv.go`, `pool.go`); design and semantics in the package's `SPEC.md`.

**Semantics:**
- NHWC, batch 1. Activations int8 asymmetric (scale, zero point); weights int8 symmetric, per output channel; bias int32 in accumulator scale. Conv weights OHWI, depthwise `[kh][kw][C]`, FC `[out][in]`.
- `acc = bias + Σ (x − zpIn)·w` in int32 (wraps on overflow; the caller bounds K). Padded taps contribute 0.
- Requantisation `Requant{Mult (Q31, [2^30, 2^31)), Shift ∈ [−31, 30]}`, real multiplier M' = Mult·2^(Shift−31): `y = clamp(round_half_away(acc·Mult / 2^(31−Shift)) + zpOut, actMin, actMax)`, computed exactly in int64 with **one** rounding. TFLite's two-step SRDHM + RoundingDivideByPOT can differ by 1 LSB on rare ties; a TFLite-compatible mode is a TODO.
- Same padding as TensorFlow (out = ⌈in/stride⌉, pad before = total/2). Pools count only valid cells; `AvgPool2D` rounds the mean of the stored int8 values half away from zero (TFLite behaviour).
- `ActMin`/`ActMax` zero values clamp everything to 0 — set them with `ActNone`/`ActReLU`/`ActReLU6`.

## Completeness review

- Complete: no — no TFLite-compatible requant mode, no dilation, no depthwise multiplier > 1, no batched GEMM (1×1 conv covers per-pixel GEMM), no saturating accumulators.
- Tested: seeded random tests against an independent exact reference (dequantise to big.Float, compute, quantise with M', round half away, clamp): Requant.Apply 1 863 720 outputs over every shift; FullyConnected 2000 cases; Conv2D 3000; DepthwiseConv2D 3000 (half 3×3, stride 1/2); MaxPool2D and AvgPool2D 3000 each; ReLU/ReLU6 1000 — **0 mismatches**. Hand-computed cases for each op, padding geometry with non-zero input zp, ±128/127 zero points, and every error path. A deliberately broken rounding was caught by the reference tests. `testing.AllocsPerRun` 0 for every kernel. TinyGo 0.43 `-print-allocs=qi8` is empty for `-target=xiao-esp32s3` and `-target=drivers/unitv/target/unitv.json`.
- Host speed (Apple M5): conv 3×3 ≈ 1.0–2.1 G MAC/s, 1×1 conv 1.4 G, depthwise 3×3 ≈ 0.6 G, FC 1024×128 2.2 G, Dot 2.5 G MAC/s.
- Correct: yes against the reference.

## TODOs

- [ ] ESP32-S3 PIE inner loops (dot, conv) behind the `esp32s3` tag, bit-identical to these kernels on the chip (ER-NN-KERNELS N2) — x/math/primitive/qi8 #p2
- [ ] TFLite-compatible two-step requantisation mode for imported TFLite models — x/math/primitive/qi8/quant.go #p2
- [ ] Depthwise channel multiplier > 1 and dilation — x/math/primitive/qi8/conv.go #p3

## Links

- Module: [[primitive-kernels]] · [[quantised-dtypes-and-fixed-point]] · [[W9-go-nn-framework]]
