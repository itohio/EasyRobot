---
type: architecture
status: draft
tags: [easyrobot, architecture, k210, kpu, nn, compiler]
workstream: [K210-nn]
source: [drivers/kpu/, drivers/kpu/kmodel/, x/math/nn/qnet/, x/math/nn/qnet/tfl/, x/math/primitive/qi8/]
evidence: "KPU format, Go arithmetic model bit-exact with nncase's simulator, silicon within ~1 LSB per layer"
---
# KPU compiler: TFLite / qnet graphs → K210 KPU layers (design)

Owner decision (2026-11-10): models come as int8 TFLite files or as our own `qnet` graphs, not as nncase kmodels. nncase stays a host-side test oracle only. This note designs a Go compiler that lowers a `qnet.Graph` to KPU layer arguments, run by `drivers/kpu`. The hardware facts it relies on are in [[k210-kpu]].

## Goal and non-goals

- **Goal:** any `qnet.Graph` (from `tfl.Load` or built in Go) runs on the K210, with the KPU executing every layer it can and qnet's CPU kernels the rest, in one plan.
  - Output within 1 LSB per layer of qnet's CPU reference for KPU-run layers.
  - No allocation per frame.
- **Non-goals:**
  - writing kmodel files;
  - training;
  - float models (int8 only, as `tfl` loads them);
  - the K230 (different NPU).

## Input

- `qnet.Graph` (`x/math/nn/qnet`) is the only input. Its layers are plain data: geometry, `InQ`/`OutQ` (real = Scale·(q − ZP), int8 activations), int8 weights (symmetric, per-output-channel `Requant`), int32 bias, fused `ActMin`/`ActMax`.
- TFLite int8 files arrive through `qnet/tfl.Load`, unchanged: the compiler never sees TFLite.
- qnet already has an accelerator seam: `qnet.Backend` (`Claim` a chain of layers, `Run` it from `src` to `dst`, NHWC int8). The KPU compiler is a `Backend` built ahead of time from the graph:

  ```go
  b, err := compile.Compile(&g, kpu.New) // lowering, AI-SRAM plan, weight blobs; kpu.NewEmulated on a host
  r, err := qnet.NewRunner(&g, arena, b) // qnet plans KPU segments and CPU layers together
  ```

## Lowering one layer

### Which layers the KPU runs

| qnet layer | KPU | Notes |
|---|---|---|
| Conv2D 1×1 or 3×3, stride 1, SAME | yes | |
| DepthwiseConv2D 3×3 (multiplier 1), stride 1, SAME | yes | |
| stride 2, `Pad(1,1)` + 3×3 VALID or 1×1 VALID | yes: conv + left-top 2×2 sampler | windows centred on even pixels. A stride-2 **depthwise** runs at full resolution (`Fuse.Full`) and the next KPU layer samples (`Fuse.Subsample`): the KPU does not pool in depthwise mode |
| stride 2, TF SAME 3×3 (even H and W) | yes, exact, no extra KPU work but a sampler | windows centred on odd pixels. The conv (regular or depthwise) runs at full resolution (`Fuse.Full`); the next KPU layer reads from one row down with the right-top sampler (`Fuse.OddSample`). That reader is the next 1×1 conv when there is one, else an identity 1×1 conv (`compile.Sampler`, exact: weights 1 and multiplier 1). The official MobileNetV1 exports need one Sampler, after the stem |
| MaxPool2D / AvgPool2D 2×2 or 4×4, stride = kernel, directly after a regular conv | fused into that conv | `pool_type`; not after a depthwise conv (CPU). The KPU's average rounds half up on the uint8 values (silicon, K9g), which is TFLite uint8's rounding (`TiesUp`) |
| everything else (Add, Concat, FC, Mean/GAP, Softmax, Logistic, Mul, resizes) | no | CPU (qnet kernels) |

### Integer mapping

The activation and weight mapping is exact:
- **KPU tensors are uint8:** x′ = q + 128. Weights: w′ = q_w + 128.
- **The KPU accumulates** Σ w′x′ + (arg_x·Σx′ >> shr_x) + (arg_w·Σw′ >> shr_w) + arg_add·n (n = taps × IC, or taps for depthwise).
- **TFLite needs** Σ (q_w − zp_w)·(q_x − zp_x) = Σ (w′ − 128 − zp_w)(x′ − 128 − zp_x), with zp_w = 0 for symmetric int8 weights and zp_w = uint8 zero point − 128 for TF's uint8 "quant" models. So:

  | Field | Value |
  |---|---|
  | arg_x | −(128 + zp_w) (shr_x 0) |
  | arg_w | −(128 + zp_x) (shr_w 0) |
  | arg_add | (128 + zp_w)·(128 + zp_x) per tap |
  | pad_value | zp_x + 128 |

  - arg_add is scaled by the tap count where nncase does (k·k folded in for 3×3; × IC by the hardware for conv).
  - pad_value is the real zero, which is what SAME padding needs.
  - All are integers: **the accumulator equals TFLite's**, if the KPU's accumulator doesn't overflow (its width is unknown, an open question).

Requantisation is approximate:
- **Batch norm then activation:** BN computes (acc·mul >> s) + add per channel (mul 24-bit signed, s ≤ 15). The activation then (y − start)·y_mul carry-shifted by up to 20 bits.
- **Per-channel scale:** TFLite's per-channel multiplier M_c = s_w,c·s_x/s_out goes into BN mul_c. The int32 bias goes into BN add_c (bias·mul_c, shifted). The remaining 2^k scale goes into the activation's y_mul/shift (nncase's "postMul" upshift, which keeps precision when M_c is small).
  - So per-channel weight scales are supported. nncase rc5 instead folds them into the weights (each channel stretched to the tensor's range before one per-tensor quantisation, see [[k210-kpu]]). TFLite int8 weights already carry per-channel scales, so the compiler keeps the weights as they are and puts the scales in BN.
- **Rounding:** BN shifts with a floor; the activation's `carry_shift` rounds a positive value half up, and gives a negative value whose dropped bit is set floor − 1 (rule in [[k210-kpu]]). TFLite rounds once (rounding doubling high multiply plus a rounding shift). So results can differ by 1 LSB from qnet's CPU output. The bar is ≤ 1 LSB per layer, not bit-exactness. Exact negative halves: the carry shift rounds them up (toward zero), TFLite and qnet away from zero. For layers imported from TFLite (`qi8.RoundTFLite`) the table splits at zero, so a negative tie rounds away from zero as TFLite's does; RoundExact layers keep the single half-up segment, where the split measured worse because BN floors first (measured: RoundExact right 1×, wrong 14×; RoundTFLite right 12×, wrong 6×). The split needs a representable negative output (clamp below the output zero point): the official uint8 MobileNet's ReLU6 layers have none, its logits layer does.
- **Output zero point and clamps:**
  - zp_out + 128 goes into the active segment's result_bias byte.
  - ActMin/ActMax (ReLU, ReLU6, other fused clamps) become flat segments (mul 0, bias = the clamp in uint8).
  - LeakyReLU uses two sloped segments.
  - All within the 16-segment table.

### Weight and table blobs

- **Weights:** uint8 in the emulator's order: OIHW for conv, `oc·9 + ky·3 + kx` for depthwise; 128-aligned. qnet stores OHWI, so the compiler transposes once.
- **Loads:** split into loads of ≤ 30 KiB (`load_time`, `o_ch_num_coef`), as nncase does.
- **Tables:** BN at 128-byte alignment (8 B per channel); activation table at 256-byte alignment (18 × u64).
- All blobs live in one compiler-owned, 128-aligned buffer below 4 GiB (the KPU fetches them by 32-bit address).

## Planning a chain

- **Segments:** a KPU segment is a maximal chain of KPU-able single-input layers.
  - Its input is uploaded once: int8 NHWC → uint8 CHW, then `UploadTo` in the first layer's layout.
  - Its output is read back once (`DownloadFrom`, then uint8 CHW → int8 NHWC).
  - Intermediate tensors stay in AI SRAM.
- **Conversion at boundaries:** the CHW↔NHWC transpose and the ±128 shift happen only there, on the CPU. [INFERENCE: cost similar to K8c-1's upload and read-back, a few hundred µs at QVGA; measured in K9e.]
- **AI-SRAM plan:** 2 MiB = 32,768 lines. A tensor takes `row_length·H·⌈C/group⌉` lines. Live tensors in a chain are only the current input and output, so a two-region ping-pong (nncase's top/bottom allocation) suffices. The plan is computed at compile time and checked: every layer's input + output ≤ 2 MiB.
- **Row-band tiling** applies when input + output > 2 MiB. Example: MobileNetV2's first expanded depthwise at QVGA.
  - The layer is split into horizontal bands with a 1-row halo for 3×3, each band a separate layer argument with its own image size and addresses, run in turn.
  - Band outputs are read back to main memory and re-uploaded as the next layer's input, or written into a tensor whose layout the next layer reads.
  - This is the most complex part and gets its own milestone.
- **CPU fallback:** layers the KPU can't run (Add, Concat, FC, GAP/Mean, Softmax, Logistic, Mul, resizes) run on qnet's CPU kernels between KPU segments. FC can alternatively run as a 1×1 KPU conv on a 4×4 zero-padded map (nncase's head), chosen by an option once measured.

## Validation

1. **Per layer on the host:** each lowered layer through `kmodel.ConvParams.Emulate` (bit-exact with nncase's simulator, K8c, except average pools, which round half up as the silicon does, K9g) against qnet's `RunLayer` on the same int8 input, ≤ 1 LSB, over random shapes, quantisations, zero points, clamps and pools.
2. **Whole graph on the host:** `kpu.NewEmulated`-style engine plus qnet's runner, layer by layer through `OnLayer`, against qnet's CPU runner.
3. **On the board:** the UnitV against the host emulation, layer by layer (K8c showed the silicon within about 1 LSB per layer of the emulator).
4. **nncase as an oracle only:** for a float model, the compiler's per-layer outputs and nncase's (both on the same input) should agree in distribution. They can't match bit-for-bit, because the quantisation differs.

## Training and export rules for the owner's models

- **Stride 2:** TF SAME and `Pad(1,1)` + VALID both run exactly, on even sizes. SAME on an odd size is refused.
- **Kernels** 1×1 and 3×3 only; depthwise multiplier 1.
- **Sizes:** W ≤ 1024, H ≤ 256 per KPU layer input (silicon, see [[k210-kpu]]), C ≤ 1024. A full-resolution stride-2 layer's input is the full image, so 256 rows is also the limit for the input image until row-band tiling (K9g). nncase rejects depthwise outputs smaller than 4×4: whether that is a hardware limit is open question 4.
- **Activations:** anything piecewise-linear in 16 segments: ReLU, ReLU6, clamps, LeakyReLU, hard-sigmoid/hard-swish approximations.
- **Residual Add and Concat** run on the CPU and cost a read-back plus re-upload each. Prefer V1-style plain chains for KPU efficiency, or keep adds few.

## Milestones (each with its check)

**First end-to-end target** (owner, 2026-11-10, for the recorder firmware):
- MobileNetV1 int8 TFLite at **256×256**, cut at a stated layer: by default the last pointwise conv `conv_pw_13`, 8×8×(1024·α).
- α = 0.25 at first, until the owner's own model exists: an official int8 MobileNetV1-0.25 export (TF SAME; `vww_96_int8.tflite`, resized to 256×256 by `kmodelc -input`) or a local Pad(1,1)+VALID one.
- Its feature tensor, with its quantisation, is what the recorder stores.
- At 256×256 the largest activation (128×128×16 for α = 0.25, 256 KiB) plus its input fits AI SRAM, so the target needs no row-band tiling.

| Step | Scope | Check |
|---|---|---|
| K9a | `kmodel.Builder`, BN/activation encoders and `Args` setters (the compiler's output seam); `Check` validates `Outputs`; doc fixes | ncc's kmodels rebuilt byte-identical; round-trip tests; mutation-checked |
| K9b | Lower one Conv2D/Depthwise (1×1, 3×3, stride 1, SAME) from `qnet.Layer` to Args + blobs | host: `Emulate(lowered)` vs `qnet.RunLayer` ≤ 1 LSB on ≥ 1,000 random layers; mutation-checked tests |
| K9c | `compile.Lower` with `Fuse{Pad, Act, Pool}`: stride 2 via Pad(1,1)+VALID (or 1×1 VALID) + left-top, fused 2×2/4×4 max/average pools, fused ReLU/ReLU6 clamps | as K9b on random chains (≤ 1 LSB; the average pool rounds half up as the silicon does, K9g), plus TF SAME stride 2 rejected |
| K9d | `compile.Compile(g, kpu.New \| kpu.NewEmulated)` → `qnet.Backend`: segments, AI-SRAM plan, int8 NHWC ↔ uint8 CHW at boundaries | host: MobileNetV1-0.25 256×256 built with Pad(1,1)+VALID from `qnet.Layer` values (qnet's own builder uses TF SAME) + `SyntheticParams`, cut at `conv_pw_13`, one KPU segment, per layer ≤ 1 LSB vs qnet; `Run` allocates nothing |
| K9e | The same graph on the UnitV | board vs host emulation per layer ≤ 1 LSB; ms/frame vs the K210 CPU at 256×256 |
| K9 SAME | TF SAME stride 2 (`Fuse.Full` + `Fuse.OddSample`, `compile.Sampler`); `compile.Kmodel`/`Cut`/`Resize`; `x/marshaller/kmodel` and `cmd/kmodelc` (TFLite → kmodel) | host: qnet's MobileNetV1-0.25 (TF SAME) one segment, per layer ≤ 1 LSB, Samplers exact; board: per layer equal to `Emulate`, final equal to the CPU runner |
| K9f | **End-to-end target:** the official uint8 TFLite MobileNetV1-0.25 (224 export, resized to 256×256; asymmetric weights through `Layer.WZP`) through `tfl.Load` → KPU, cut at `conv_pw_13` | done on still images: per layer ≤ 1 LSB vs qnet, which equals LiteRT; final features vs LiteRT cosine 0.992–0.997 on the host, 0.9925 on the board. Open: camera frames, and class agreement (needs the uncut model's classifier; K8c-3's deferred check) |
| K9g | Row-band tiling; CPU Add/Concat between segments; FC-on-KPU option; LeakyReLU tables (two sloped segments); MobileNetV2 at QVGA | per layer ≤ 1 LSB vs CPU; fits 2 MiB |

## Open questions (resolved by measurement in the milestones)

1. The KPU accumulator width: does a 3×3×1024 conv of extreme values overflow? (K9b: probe on the board.)
2. What `first_stride`, `bypass_conv`, `full_add`, `ram_flag` and `kernel_offset` do. nncase leaves them 0. `first_stride` may be a true stride-2 path. (K9c: probe on the board.)
3. ~~TF-SAME stride 2 on the first layer via a one-pixel input shift at upload.~~ Resolved: any SAME stride-2 layer runs exactly as full resolution + right-top sampler from one row down (pool_type 6 checked on silicon). An upload with pad rows is impossible at 256×256 anyway (258 rows exceed the 256-row limit).
4. The ≥ 4×4 depthwise output limit: is it hardware or nncase? (K9c: probe on the board.)
5. Whether BN's floor shift and the activation's 36-bit compare explain the ±1 LSB silicon difference. If so, the emulator can model the hardware exactly. (K9b, from the K8c per-layer data.)
6. Hand-made probe layers (`kpupool`, `kpushape`: BN mul 2¹⁰ shift 15, activation shift 4) differ from `Emulate` on 2–3 % of outputs on silicon, while every compiler-lowered layer matched exactly. Which BN/activation settings the emulator models inexactly is open (K9g).
