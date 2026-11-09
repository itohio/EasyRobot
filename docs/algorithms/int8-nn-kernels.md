---
type: algorithm
status: tested
tags: [easyrobot, nn, int8, quantisation, kernels, embedded]
workstream: W9
module: "[[primitive-kernels]]"
summary: "Generic int8 NHWC kernels (dot, FC, conv, depthwise, pools, ReLU/ReLU6) with exact single-rounding per-channel requantisation; bit-exact reference for SIMD paths; allocation-free."
source: [x/math/primitive/qi8/quant.go, x/math/primitive/qi8/act.go, x/math/primitive/qi8/shape.go, x/math/primitive/qi8/dot.go, x/math/primitive/qi8/conv.go, x/math/primitive/qi8/pool.go, x/math/primitive/qi8/layer.go, x/math/primitive/qi8/impl_generic.go, x/math/primitive/qi8/impl_esp32s3.go, x/math/primitive/qi8/pie_esp32s3.c]
tests: [x/math/primitive/qi8/reference_test.go, x/math/primitive/qi8/kernels_test.go, x/math/primitive/qi8/bounds_test.go, x/math/primitive/qi8/layer_test.go, x/math/primitive/qi8/bench_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/primitive/qi8 → ok (2026-11-09): 0 mismatches vs an exact big.Float float-then-quantise reference over 15 310 random cases (≈2.5 M outputs); AllocsPerRun 0 for every kernel; tinygo 0.43 -print-allocs=qi8 empty for xiao-esp32s3 and the UnitV (K210) target (ER-NN-KERNELS N1)"
---
# Int8 NN kernels (generic reference)

**What it does:** the generic int8 inference kernels of the NN layer (ER-NN-KERNELS N1): `Dot`, `FullyConnected`, `Conv2D` (any kernel, stride, same/valid padding), `DepthwiseConv2D` (channel multiplier 1), `MaxPool2D`, `AvgPool2D`, `ReLU`, `ReLU6`, plus per-channel requantisation (`Requant`, `QuantizeMultiplier`) and fused-activation clamp helpers. Pure Go, no allocation, TinyGo-buildable. It is also the bit-exact reference for the ESP32-S3 PIE kernels (N2) and the K210 CPU path.

**Where:** `x/math/primitive/qi8` (`quant.go`, `act.go`, `shape.go`, `dot.go`, `conv.go`, `pool.go`, `layer.go`, `impl_generic.go`); design and semantics in the package's `SPEC.md`.

**Two tiers (since review R3):** the free functions (`Conv2D`, …) are the always-generic exported **reference**. Prepared layer types (`Conv2DLayer`, `DepthwiseConv2DLayer`, `FCLayer`, `MaxPool2DLayer`, `AvgPool2DLayer`, `ReLULayer`, `ReLU6Layer`, all implementing `Op{InShape, OutShape, Run}`) validate once in `Prepare` (same errors as the free functions) and run many times; `Run` checks only slice lengths and returns `ErrNotPrepared` if fields changed after `Prepare`. Their inner loops go through a build-tag seam (`impl_generic.go` today; an `esp32s3 && !qi8generic` file adds PIE) with an optional caller-provided `Scratch` (`ScratchLen()`) for aligned/repacked weights; `Backend()` names the build. Contract: per-op CPU backends are byte-identical to the reference; whole-model offload (KPU) is outside this interface and compared within a tolerance. Tests: 7000 seeded cases / 1 027 872 outputs prepared vs free functions, 0 mismatches; 35-row Prepare error table; field-mutation detection; AllocsPerRun 0 for Prepare and Run; `-tags qi8generic` and 32-bit (`GOOS=linux GOARCH=arm go vet`) builds.

**ESP32-S3 PIE backend (ER-NN-KERNELS N2):** `impl_esp32s3.go` + `pie_esp32s3.c` (`esp32s3 && !qi8generic`). Conv2D and FullyConnected gather each receptive field into a 16-byte-aligned patch (padded cells = zpIn, zero tail to a multiple of 32) and run `qi8_pie_dot_rows` against weight rows repacked at `Prepare` (fused `ee.vmulas.s8.accx.ld.ip`, software-pipelined, `loopnez`); the zero point is folded as `bias − zpIn·Σw` (exact in wrapping int32). Weights may be packed in place (`W` aliasing the aligned scratch start) so only one copy exists. `init()` sets CPENABLE bit 3 (PIE is coprocessor 3; TinyGo enables only the FPU). Pools and activations are still generic (depthwise: next paragraph). On the chip (XIAO ESP32-S3, 240 MHz): 17 000 random Conv2D/FC cases, 2.5 M outputs per path (incl. in-place pack and misaligned scratch), **0 mismatches** vs the free functions. Conv 3×3 2.44 cycles/MAC (generic 19.7), pointwise 3.4–3.8 (generic ≈ 22). Benchmarks: `cmd/nnbench`.

**PIE depthwise (ER-NN-PIPELINE step 1):** `qi8_pie_dw_px` accumulates 16 channels per lane group in QACC (`ee.vmulas.s8.qacc`). QACC lanes are 20 bits and **saturate** at ±(2^19−1) (measured on the chip: 32·16384 clips to 0x7FFFF), so taps are summed in chunks of ≤ 31 and added in wrapping int32 — exact for any kernel size. QACC is spilled with `ee.st.qacc_{l,h}.{l.128,h.32}`: lanes 0–7 in QACC_L, 8–15 in QACC_H, packed as little-endian 20-bit fields (layout measured on the chip). When C is a multiple of 16 and the input is 16-byte aligned, groups load straight from the input through a per-pixel tap-pointer list (padded taps point at a zero-point row); otherwise each group is staged (16 bytes per tap). Bias folding `bias − zpIn·Σw` as for conv; requantisation runs in C with the same single exact rounding. On the chip: 11 200 random depthwise cases (kernels up to 7×7, non-square, strides 1–3, SAME/VALID, channels 1–33, aligned and misaligned inputs) within a 27 200-case run, 0 mismatches; depthwise 50.9 → 26.6 cycles/MAC in MobileNetV1-0.25.

**Semantics:**
- NHWC, batch 1. Activations int8 asymmetric (scale, zero point); weights int8 symmetric, per output channel; bias int32 in accumulator scale. Conv weights OHWI, depthwise `[kh][kw][C]`, FC `[out][in]`.
- `acc = bias + Σ (x − zpIn)·w` in int32; the reduction length is limited to `MaxTaps` = 65 793 so the sum cannot wrap (the bias term is the caller's responsibility). Every size is overflow-checked against `MaxElements` = 2^30 (fits a 32-bit int); violations return `ErrShape` before any slice is touched (`TestHugeShapesRejected`, `TestReductionLengthBound`, red first: a wrapped shape used to pass the length checks and `MaxPool2D` panicked). Padded taps contribute 0.
- Requantisation `Requant{Mult (Q31, [2^30, 2^31)), Shift ∈ [−31, 30]}`, real multiplier M' = Mult·2^(Shift−31): `y = clamp(round_half_away(acc·Mult / 2^(31−Shift)) + zpOut, actMin, actMax)`, computed exactly in int64 with **one** rounding. TFLite's two-step SRDHM + RoundingDivideByPOT can differ by 1 LSB on rare ties; a TFLite-compatible mode is a TODO.
- Same padding as TensorFlow (out = ⌈in/stride⌉, pad before = total/2). Pools count only valid cells; `AvgPool2D` rounds the mean of the stored int8 values half away from zero (TFLite behaviour).
- `ActMin`/`ActMax` zero values clamp everything to 0 — set them with `ActNone`/`ActReLU`/`ActReLU6`.

## Completeness review

- Complete: no — no TFLite-compatible requant mode, no dilation, no depthwise multiplier > 1, no batched GEMM (1×1 conv covers per-pixel GEMM).
- Tested: seeded random tests against an independent exact reference (dequantise to big.Float, compute, quantise with M', round half away, clamp): Requant.Apply 1 863 720 outputs over every shift; FullyConnected 2000 cases; Conv2D 3000; DepthwiseConv2D 3000 (half 3×3, stride 1/2); MaxPool2D and AvgPool2D 3000 each; ReLU/ReLU6 1000 — **0 mismatches**. Hand-computed cases for each op, padding geometry with non-zero input zp, ±128/127 zero points, and every error path. A deliberately broken rounding was caught by the reference tests. `testing.AllocsPerRun` 0 for every kernel. TinyGo 0.43 `-print-allocs=qi8` is empty for `-target=xiao-esp32s3` and `-target=drivers/unitv/target/unitv.json`.
- Host speed (Apple M5): conv 3×3 ≈ 1.0–2.1 G MAC/s, 1×1 conv 1.4 G, depthwise 3×3 ≈ 0.6 G, FC 1024×128 2.2 G, Dot 2.5 G MAC/s.
- Correct: yes against the reference.

## TODOs

- [x] ESP32-S3 PIE inner loops (dot, conv, FC) behind the `esp32s3` tag, bit-identical on the chip (ER-NN-KERNELS N2) — x/math/primitive/qi8/impl_esp32s3.go #p2
- [x] PIE depthwise conv (per-lane QACC accumulation across channels) — x/math/primitive/qi8/impl_esp32s3.go #p2
- [ ] TFLite-compatible two-step requantisation mode for imported TFLite models — x/math/primitive/qi8/quant.go #p2
- [ ] Depthwise channel multiplier > 1 and dilation — x/math/primitive/qi8/conv.go #p3

## Links

- Module: [[primitive-kernels]] · [[quantised-dtypes-and-fixed-point]] · [[W9-go-nn-framework]]
