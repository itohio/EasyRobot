---
type: architecture
status: active
tags: [easyrobot, architecture, k210, kpu, nn]
workstream: [K210-nn]
source: [drivers/kpu/, drivers/kpu/kmodel/]
evidence: "~/src/docs/evidence/ER-K210-K8.md (outside the repo): tiny1/2/3 on the UnitV vs nncase v0.1.0-rc5's simulator; field values below decoded from those kmodels"
---
# K210 KPU: layer format and arithmetic

What the KPU executes, field by field, so that a compiler other than nncase (for example from TFLite or `qnet` graphs) can emit layers for `drivers/kpu`. Sources:
- **kendryte-standalone-sdk V0.5.6:** `lib/drivers/kpu.c`/`kpu.h`, the kmodel v3 runtime.
- **nncase v0.1.0-rc5:** `src/NnCase.Converter.K210/Converters/Layers/K210Conv2d.cs` (codegen, quantisation and emulator), `K210Helper.cs`, `Stages/Quantize/QuantizationContext.cs`. All Apache-2.0, read at the tags.
- **The UnitV runs** in the evidence note.

"nncase" below means rc5's behaviour, which is not necessarily the hardware's.

## Memory

- **KPU RAM:** 2 MiB, addressed in 64-byte lines (15-bit line addresses), at 0x8060_0000 (cached) and 0x4060_0000 (uncached; the runtime uses this one).
- **Registers:** at 0x4080_0000 (`kendryte.KPU`): `LAYER_ARGUMENT_FIFO`, `INTERRUPT_{STATUS,RAW,MASK,CLEAR}` (bit 0 calc_done, bit 1 FIFO almost empty, bit 2 almost full), `FIFO_THRESHOLD`, `FIFO_DATA_OUT`, `FIFO_CTRL`, `EIGHT_BIT_MODE`.
- **Clock:** PLL1 / (`CLK_TH0.AI_GCLK` + 1), enabled by `CLK_EN_PERI.AI_CLK_EN`. PLL1 is off after a TinyGo boot; `drivers/kpu` starts it at 398.67 MHz.
- **Weights, batch-norm and activation tables** stay in main memory. The KPU fetches them by 32-bit address (`para_start_addr`, `bwsx_base_addr`, `active_addr`). nncase aligns them to 128, 128 and 256 bytes.

### Tensor layout in KPU RAM

A CHW uint8 tensor of width W, height H:

| W | Channels per line (group) | Bytes per channel slot | Lines per row (`row_length`) |
|---|---|---|---|
| ≤ 16 | 4 | 16 | 1 |
| ≤ 32 | 2 | 32 | 1 |
| > 32 | 1 | 64 | ⌈W/64⌉ |

Byte (c, y, x) of a tensor at line address A is at `A·64 + ⌊c/group⌋·row_length·H·64 + (c mod group)·64/group + y·row_length·64 + x`.
- The input side of a layer gives these as `kernel_calc_type_cfg` (`coef_group`, `row_switch_addr` = row_length, `channel_switch_addr` = row_length·H); the output side as `write_back_cfg` (`wb_group`, `wb_row_switch_addr`, `wb_channel_switch_addr`).
- `kmodel.Layout` implements this; `UploadLayout` equals `kpu_upload_core`, as tested.
- A tensor takes row_length·H·⌈C/group⌉ lines.
- nncase allocates KPU RAM per layer from the top (tiny3: input at line 32,704, layers ping-pong between lines 0 and 32,736).

## Layer argument: 12 × u64, written in order to `LAYER_ARGUMENT_FIFO`

Bit positions from `kpu.h`; "nncase" is what rc5 writes; tiny3 values are given where useful.

| # | Register | Fields (bits) | Meaning / how nncase fills it |
|---|---|---|---|
| 0 | `interrupt_enabe` | int_en 0, ram_flag 1, full_add 2, depth_wise_layer 3 | int_en raises calc_done at the layer end. The runtime sets it per run; nncase leaves it 0. `depth_wise_layer` is 1 for depthwise. |
| 1 | `image_addr` | image_src_addr 0–14, image_dst_addr 32–46 | input and output line addresses |
| 2 | `image_channel_num` | i_ch_num 0–9, o_ch_num 32–41, o_ch_num_coef 48–57 | channels − 1. `o_ch_num_coef` = output channels per weight load − 1 |
| 3 | `image_size` | i_row_wid 0–9, i_col_high 10–18, o_row_wid 32–41, o_col_high 42–50 | sizes − 1; max 1024 × 512 |
| 4 | `kernel_pool_type_cfg` | kernel_type 0–2, pad_type 3, pool_type 4–7, first_stride 8, bypass_conv 9, load_para 10, dma_burst_size 16–23, pad_value 24–31, bwsx_base_addr 32–63 | kernel_type 1 = 3×3, 0 = 1×1. pool_type: see below. nncase: load_para 1, burst 15, pad_value = round(−bx) (the input's quantised zero; 128 for the n1to1 input, 0 after ReLU). bwsx = batch-norm table address. |
| 5 | `kernel_load_cfg` | load_coor 0, load_time 1–6, para_size 15–31, para_start_addr 32–63 | weights address; para_size = bytes per load; load_time = loads − 1 (nncase: ≤ 30 KiB of 8-bit weights per load) |
| 6 | `kernel_offset` | coef_column_offset 0–3, coef_row_offset 4–15 | nncase: 0 |
| 7 | `kernel_calc_type_cfg` | channel_switch_addr 0–14, row_switch_addr 16–19, coef_size 20–27, coef_group 28–30, load_act 31, active_addr 32–63 | input layout (above); load_act 1; activation table address |
| 8 | `write_back_cfg` | wb_channel_switch_addr 0–14, wb_row_switch_addr 16–19, wb_group 20–22 | output layout |
| 9 | `conv_value` | shr_w 0–3, shr_x 4–7, arg_w 8–31, arg_x 32–55 | zero-point terms; arg_w and arg_x are signed 24-bit (tiny3 layer 0: arg_w 0xC00000 = −4,194,304) |
| 10 | `conv_value2` | arg_add 0–39 | zero-point product term |
| 11 | `dma_parameter` | send_data_out 0, channel_byte_num 16–31, dma_total_byte 32–63 | send_data_out streams the output to `FIFO_DATA_OUT` (for DMA); byte counts − 1 (W·H, W·H·C). `drivers/kpu` leaves it off and reads the output from KPU RAM. |

**pool_type** (nncase emulator; stride = kernel). **Silicon: regular convs only.** In depthwise mode a non-zero pool_type gives wrong output on the UnitV (ER-K210-K9 evidence, K9e). nncase never pools a depthwise layer: it lowers a stride-2 depthwise at full resolution and puts the left-top sampler on the following 1×1 conv.

| Value | Pool |
|---|---|
| 0 | none |
| 1 | 2×2 max |
| 2 | 2×2 average |
| 3 | 4×4 max |
| 4 | 4×4 average |
| 5 | 2×2 "left-top": keep the top-left pixel, which is how nncase does stride 2 |

The conv itself is always stride 1, with SAME padding filled with `pad_value`. Output size = input size, then divided by the pool stride.

## Arithmetic (nncase's emulator)

`kmodel.ConvParams.Emulate` is a Go port of this arithmetic, and `kpu.NewEmulated` runs a whole kmodel on any host with it. Both are bit-identical to nncase rc5's simulator on tiny1/2/3 and on the 27-layer MobileNetV1-0.25 feature extractor. On the UnitV, each layer's output, computed from that layer's own KPU input, agrees with the port to within 1 LSB: 503,605 of 503,808 outputs equal, 203 at 1 LSB, none above (ER-K210-K8 evidence, K8c-2). The 1-LSB cases compound across layers, so a 27-layer output can differ from the simulator by up to about 25 LSB.

**Quantisation.** For a range [min, max]: scale s = 255/(max − min), bias b = round(min·s), and real = (q + b)/s. Weights use the same with 8 bits. The rc5 release is built `CHANNEL_WISE`: it first scales each output channel's float weights by s = min(total max / channel max, total min / channel min) (the larger of the two if either is negative), stretching them to the tensor's range, then quantises the whole tensor with one scale and bias. This reproduces all 61 weight blobs of the K8 models byte for byte (ER-K210-K9 evidence, K9b); the per-tensor rule alone reproduces none.

**Conv.** Inputs x and weights w are uint8.
- For each output pixel, with Σ over the kernel window (3×3·IC, or IC for 1×1; 3×3 per channel for depthwise):
  ```
  acc = Σ w·x + (arg_x·Σx >> shr_x) + (arg_w·Σw >> shr_w) + arg_add·IC
  ```
  For depthwise, the last term is `arg_add` without the IC factor.
- nncase: arg_x/shr_x encode bw (the weight bias), arg_w/shr_w encode bx (the input bias), both as 24-bit mantissa plus shift; arg_add = round(bw·bx·k·k). The terms expand Σ(w + bw)(x + bx).
- Out-of-image taps read `pad_value`.

**Batch norm**, per output channel (64-bit entry: norm_mul 0–23, norm_add 24–55, norm_shift 56–59):
```
y = (acc·norm_mul >> norm_shift) + norm_add
```
- nncase rc5 (the `CHANNEL_WISE` build): per channel, mul and shift encode so·2^10/(sw_c·sx) as a 22-bit mantissa with shift ≤ 15 (the MobileNet files show shift 15, sometimes 14), and add = round((bias·so − bo)·2^10). The 2^10 ("postMul") is undone by the activation's linear segment: y_mul 1024, shift 20. Folding the float bias and the output quantisation in here is how the conv bias reaches the hardware.

**Activation:** a 16-segment piecewise-linear table (16 × u64: shift_number 0–7, y_mul 8–23, x_start 24–59 as signed 36-bit; then 2 × u64 of 8 result_bias bytes):
```
seg = last i with x_start[i] ≤ y
out = clamp(carry_shift((y − x_start)·y_mul, shift) + bias[seg], 0, 255)
```
- `carry_shift` (`kpu.c` `kpu_carry_shift`): if the last bit shifted out is 1, a positive value rounds up and a negative value gets floor − 1 (−5 >> 1 → −4). Otherwise it is a floor shift.
- nncase ReLU/ReLU6/linear: segments 0–9 start at large negative values with mul 0 (output 0); segment 10 starts at 0 with mul = 1/postMul (16-bit mantissa, shift ≤ 20); segments 11–15 start near 2^35 (unused).
  - ReLU6's upper clip comes only from the output range: 6.0 quantises to 255.
  - LeakyReLU uses all 16 segments.

**Pool** (`pool_type`), then the result is written in the output layout.

## Weights blob

8-bit weights, one byte each, in the emulator's index order:

| Kind | Index of weight |
|---|---|
| 3×3 conv | `((oc·IC + ic)·3 + wy)·3 + wx` (OIHW) |
| 3×3 depthwise | `oc·9 + wy·3 + wx` |
| 1×1 conv | `oc·IC + ic` |
| 1×1 depthwise | `oc` |

With more than one load, each load holds `o_ch_num_coef + 1` output channels. The weight blob is `IC·k·k·OC` bytes (`IC·k·k` for depthwise), doubled with 16-bit weights (header flag bit 0 clear). That is not `para_size × loads`: nncase's last load is partial (MobileNet layer 26: 3 loads of 30,720 B for 65,536 B), and `Check` sizes the blob from the shape.

## Table sizes and alignment

| Table | Size | Alignment in the file (ncc) |
|---|---|---|
| layer argument | 12 × u64 = 96 B | 8 |
| weights | see above | 128 |
| batch norm | 8 B per output channel | 128 |
| activation | 16 × u64 + 16 result_bias bytes = 144 B | 256 |

`kmodel.EncodeBN` and `kmodel.EncodeAct` write the last two; `kmodel.Builder` lays a model out exactly as ncc does. Rebuilt from their parsed parts, all five ncc kmodels of the K8 evidence (tiny1/2/3, mbnet025fe, mbnet025) are byte-identical (ER-K210-K9 evidence, K9a).

## kmodel v3 container

Layout is in the `kmodel` package doc. Layer types are listed in `kpu.h` `kpu_model_layer_type_t`; K210 types start at 10240.
- **Bodies** start right after the layer table, in layer order. Each is padded with zeros so that it **ends** on a multiple of 8 in the file. The padding counts in its size.
- **K210 conv body:** flags (bit 0 = main-memory output), main-memory output address, and file offsets of the layer argument, weights, BN and activation tables. ncc places those four inside the conv's own body, right after these six words.
- **CPU layer bodies** (all u32 unless noted; addresses are byte offsets in main memory, except where marked KPU):

  | Layer | Fields in order |
  |---|---|
  | Quantize | flags, in, out, count, scale f32, bias f32 (float in → uint8 out) |
  | Dequantize | flags, in, out, count, scale f32, bias f32 (uint8 in → float out) |
  | GlobalAveragePool2D | flags, in, out, kernel size, channels (float) |
  | K210AddPadding | flags, in, out (KPU line address), channels |
  | K210RemovePadding | flags, in, out, channels (keeps byte 16·c of each channel) |
  | K210Upload | flags, in, out (KPU line address), width, height, channels |

- **CPU layers:** they work on a main-memory arena (`main_mem_usage` bytes).
- **What nncase emits, in practice:**
  - nothing for the input: the first conv reads uint8 CHW straight from KPU RAM;
  - `Dequantize` at the end;
  - for a classifier head: `Dequantize → GlobalAveragePool2D → Quantize → K210AddPadding → K210Conv2d (1×1, on a 4×4 padded map) → K210RemovePadding → Dequantize`.
- `Runner.Output(i)` panics for i outside the model's outputs; `Check` guarantees each output lies inside main memory.

## Compiler constraints learned from nncase rc5

- Only 3×3 and 1×1 kernels, stride 1. Stride 2 is the conv followed by left-top 2×2 pooling, so windows centre on even pixels (for a depthwise conv, the pooling goes on the next 1×1 conv, see pool_type). That equals TF `PAD [[1,1],[1,1]]` + VALID, not TF SAME (which centres on odd pixels for even sizes).
  - nncase rc5 accepts a stride-2 3×3 depthwise only as PAD(1,1) → depthwise VALID stride 2 → 1×1 conv, and PAD only with exactly that padding. The official MobileNetV1 TFLite therefore fails with "DepthwiseConv2d is not supported".
- Max 1024 channels (10-bit fields), width ≤ 1024, height ≤ 512, and the tensor must fit 2 MiB together with the other live tensors.
- Activations: anything a 16-segment piecewise-linear function of the BN output can express.
- **Fully connected layers:**
  - on the KPU as a 1×1 conv on a 4×4 zero-padded map (AddPadding/RemovePadding), which is how nncase runs a classifier on the KPU and what `drivers/kpu` runs. The KPU's minimum useful width is 4.
  - The kmodel format also has a float CPU layer (`KL_FULLY_CONNECTED`, `kpu_kmodel_fully_connected`), but `drivers/kpu` does not implement it, and `Check` rejects it as unsupported. The K9 compiler runs FC on qnet's CPU kernels outside the kmodel instead ([[kpu-compiler]]).
- **rc5's simulator** has no Quantize layer and is not bit-exact with the silicon. Differences grow about 1 LSB per layer (evidence ER-K210-K8).
