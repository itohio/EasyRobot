---
type: workstream
status: proposed
tags: [easyrobot, workstream, nn, tensor, tinygo, llm]
workstream: W9
---
# W9 — Go NN framework on ESP32-Xtensa, K230, native LLM, TinyGo face recognition

**Goal:** the Go NN framework (tensor math implemented in Go) should run on the ESP32 (Xtensa) with optimised kernels, support TinyGo camera and K230 neural compute work, and run a small language model (for example Qwen 0.5B).

Related tickets (planning, keep using them): ER-003 operator capabilities, ER-004 quantisation contract, ER-005/006 Xtensa, ER-007/008 K230, ER-011…014 small-LLM path, ER-015 measurement. Graph semantics and backend limits: [[source-inventory]] §3–5 (not repeated here).

## What exists

| Layer | Note | State (2026-11-09 run, darwin/arm64, go1.27.2) |
|---|---|---|
| Tensor interface + eager FP32 backend + Gorgonia graph backend | [[tensor]] | eager 112/112 tests pass; gorgonia 30 pass, `TestGraphConv2D` skipped; many gorgonia methods `panic("not implemented")` |
| FP32 BLAS/conv/activation/norm/FFT primitives | [[primitive-kernels]] | fp32 97/97, c64 30/30 pass; pure Go, no asm; qi*/qu8/bytes packages are 7-line stubs |
| Layers, losses, Sequential model | [[nn]] | layers 91 pass / 2 fail (both test defects, see below), **but value probes show most layers beyond Dense/activations are wrong or crash**: attention `Forward` always panics, positional encoding panics for seqLen ≥ 2, LSTM forward reads every gate from `gates[0:h]`, BatchNorm2D computes InstanceNorm, Conv1D kernel-grad wrong for >1 output channel, Conv2D input-grad wrong for stride tails, `Sequential.Parameters()` drops tensors (details in [[nn]]); models + nn root have no own tests |
| Root cause shared by several layers | [[tensor-views-strides]] | eager `Data()` returns the whole backing slice and ignores a view's offset (`eager/tensor.go:103-111`); ops that read `Data()` on `Slice` views see the wrong elements; plus misuse of `Slice(dst, dim, start, length)` (length passed as end) and of `Copy(src)` direction in GPT layers |
| Optimisers SGD/Adam, TrainStep, quantise-model | [[learn]] (other slice) | — |
| Type conversion / clamp / strided copy | [[type-conversion]] | x/types 27 pass; x/types/st tests do not build on arm64 (amd64-only asm) |
| Iterators, worker pool (`use_mt`) | [[iter-workers]] | iter 57, workers 19 pass; `loop_experiments` does not build on arm64 |
| Small fixed matrices / vectors / decompositions | [[mat-vec]] | mat 145, vec 61 pass (many vacuous: empty `//TODO: Add test cases` tables) |
| TFLite import (linux, CGO) | [[tflite]] | not built/run on this host |

Command: `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/tensor/... ./x/math/nn/... ./x/math/primitive/... ./x/math/mat/... ./x/math/vec/... ./x/math/shape/... ./x/math ./x/iter/... ./x/workers/... ./x/types/...` → exit 1; failures only as listed.

### The two baseline failures (investigated)

- `TestDense_NumericalGradientCheck` (`x/math/nn/layers/dense_comprehensive_test.go:253`): **test defect.** Forward difference with `epsilon=1e-5` in float32. Loss = sum of outputs ≈ 6.1; float32 spacing near 6 is 4.77e-7, so a 1e-5 change rounds to 22 ulp = 1.049e-5 → reported numerical gradient 1.049042 vs analytical 1.0 (and 3.051758 vs 3.0). The analytical gradients equal the inputs (1, 2, 3), which is the exact gradient of sum(Wx+b) w.r.t. W. Fix: central difference with eps ≈ 1e-2…1e-3, or compute the loss in float64. Detail: [[dense-layer]].
- `TestGPTEmbedding_ErrorCases` (`gpt_embed_test.go:82`): **stale test assertion.** The implementation accepts 1-D and 2-D input and says "input must be 1D or 2D, got 3D" (`gpt_embedding.go:63`); the test expects "input must be 2D". But the 1-D path the message advertises is **wrong in the implementation**: `gpt_embedding.go:145–147` calls `Slice(nil, 0, tokenID, tokenID+1)` where the 4th argument is a *length* (`eager/tensor.go:592`), and `embedSlice.Copy(outputSlice)` copies the output *into* the embedding table (`Copy(src)` copies src into the receiver, `eager/tensor.go:300`). Detail and probe: [[gpt-token-embedding-layer]].

## (a) TinyGo / ESP32-Xtensa readiness — static reading only (nothing built with TinyGo)

Counts over non-test files (regex scan, 2026-11-09):

| Package | LOC | `reflect` import | `unsafe` | `go` stmt | `.(type)` switches | `float64` tokens | `panic(` | `make(` |
|---|---|---|---|---|---|---|---|---|
| tensor/eager | 7926 | 0 | 1 (`tensor.go:80`) | 0 | 82 | 151 | 403 | 23 |
| tensor/types | 1169 | 0 | 0 | 0 | 3 | 44 | 1 | 1 |
| nn/layers | 6075 | 0 | 0 | 0 | 0 | 24 | 0 | 51 |
| primitive/fp32 | 7694 | 0 | 0 | 0 | 0 | 5 | 23 | 17 |
| x/types (+st) | 6699 | 0 | 0 | 0 | 175 | 283 | 0 | 7 |
| x/workers | 559 | 0 | 0 | 2 (`worker_pool.go:135,197`) | 0 | 0 | 0 | 3 |

Eager panics (403) cover every unsupported dtype/shape path; `float64` counts include API scalars.

Findings:

1. **Production code imports testify.** `x/math/nn/layers/pooling_test_edgecases.go` is not a `_test.go` file; it defines `TestMaxPool2D_EdgeCases` etc. and imports `testing` + testify, so `go list -deps ./x/math/nn/layers` (non-test) contains `stretchr/testify`, `gopkg.in/yaml.v3`, `davecgh/go-spew`. This is the first thing that would break or bloat a TinyGo build of `nn/layers`.
2. **float64 in the core API.** `Tensor.At/SetAt`, scalar ops, eps and alpha arguments are float64 (`tensor/types`, 44 tokens; eager 151). ESP32 (LX6) and ESP32-S3 (LX7) have a single-precision FPU only [INFERENCE: Espressif datasheets, not checked offline]; every float64 op is soft-float. Kernels themselves run on `[]float32`.
3. **`any`-typed storage with type switches.** `Tensor.Data() any`; eager dispatches with 82 type switches and generic helpers in `x/types` (175 switches). TinyGo supports generics and type switches, but this costs code size and per-call overhead; eager MatMul/Conv only implement `[]float32` and panic otherwise (e.g. `eager/linalg.go:30–41`).
4. **Allocation in hot paths.** All ops take `dst`; `dst == nil` allocates. Layers pre-allocate outputs in `Init`, but kernels such as Conv2D allocate im2col/GEMM workspaces per call (declared in `primitive/fp32/PERFORMANCE_REPORT.md` §Next steps; see [[conv2d-kernel]]). TinyGo's GC is conservative and slow; a workspace/arena contract is needed.
5. **Concurrency is opt-in.** Default build uses `x/iter/st` (no goroutines); `-tags use_mt` pulls `x/workers` (goroutines, `sync.Pool`, `runtime.NumCPU`). Good for TinyGo.
6. **No Xtensa or RISC-V code.** The only assembly is amd64 (`x/iter/loop_experiments/*_amd64.s`, `x/types/st/convert_asm_amd64.s`), and it lacks generic fallbacks, which already breaks arm64 builds (see [[W2-code-optimisation]]).
7. **`fmt`/`panic(fmt.Sprintf…)`** on error paths throughout (eager 7 files import fmt, layers 11). Works in TinyGo but adds size.
8. **External math dependency** `github.com/chewxy/math32` everywhere; TinyGo/Xtensa compatibility not checked [unverified].
9. **No int8 compute path.** Dtype tags INT8/UINT8/INT48/FP16 exist but kernels are FP32; no scale/zero-point; `primitive/qi*` are empty stubs. See [[quantised-dtypes-and-fixed-point]].
10. **Int-size files gated by explicit GOARCH lists.** `x/types/st/{convert,cast}_i32.go`, `x/types/internal/constraints_i32.go`, `x/types/{value,clamp}_i32.go` are `386||arm||mips||mipsle`; the `_i64` twins list `amd64||arm64||…||riscv64||s390x`. Any other GOARCH gets neither: `GOOS=js GOARCH=wasm CGO_ENABLED=0 go build ./x/types/...` fails (`x/types/st/cast.go:82: undefined: elemConvertToInt`, `convert.go:53`, `convert_ops.go:44`…), so the whole tensor/nn stack does not build for wasm. A TinyGo target only builds if it reports one of the listed GOARCHs (TinyGo reports `arm` for its Xtensa targets [INFERENCE, unverified]). Fix: one file using `strconv.IntSize`/`math.MaxInt`.

Cross-compile check (2026-11-09, compile only, `CGO_ENABLED=0 go build` of `tensor/types`, `tensor/eager`, `tensor`, `primitive/...`, `nn/...`, `mat`, `vec`, `iter/...`, `workers/...`, `types/...`): **linux/riscv64, linux/arm, linux/386 → only `x/iter/loop_experiments` fails** (missing asm bodies `bce_experiments.go:317,318,374,375`, `matmul.go:5`); **js/wasm → also `x/types/st` fails** (above).

Verdict: the numeric core is pure Go, CGO-free and single-threaded by default, which is the right starting point; the blockers are the testify leak, float64 API, per-call allocation and missing int8/SIMD kernels. Nothing is verified on TinyGo.

## (b) K230

No K230/KPU/nncase code exists ([[source-inventory]] §7). K230 runs Linux on its big RISC-V core [INFERENCE], so the standard Go toolchain (`GOARCH=riscv64`) is the natural host path, not TinyGo; the NPU needs vendor nncase/kmodel tooling (ER-007). The W4 plan (feature tensor → SPI → ESP32) means the framework on K230 is mainly the CPU reference/fallback; the ESP32 side mainly needs tensor (de)serialisation and small heads. The numeric core cross-compiles for linux/riscv64 (compile only, see the check above); nothing ran on RISC-V.

## (c) Native Go LLM inference of Qwen 0.5B

**What exists (as code, not as working layers):** token embedding layer ([[gpt-token-embedding-layer]]: 2-D path correct by probe, 1-D path panics or overwrites the table), absolute positional encoding ([[gpt-positional-encoding-layer]]: panics for seqLen ≥ 2, non-standard frequencies), multi-head causal self-attention with a full `[maxSeqLen,maxSeqLen]` additive mask ([[gpt-attention-layer]]: `Forward` always panics — wrong `Slice` lengths and head reshape; `Backward` zeroes the gradient; dropout TODO `gpt_attention.go:328`), RMSNorm/LayerNorm kernels ([[rms-norm]], [[layer-norm]]), Softmax ([[softmax-kernel]]), GELU/Swish activations ([[activation-kernels]]), FP32 MatMul ([[tensor-matmul]]). None of the GPT layers has a value test; none can run a forward pass of a transformer block today. `Sequential` has no residual/branching, so a decoder block (x + Attn(norm(x))) cannot be expressed with the current model type either.

**Missing (source search for safetensors, gguf, tokeniz, BPE, rotary/RoPE, kv cache, SwiGLU, SiLU, grouped-query over `x/` and `cmd/` found nothing):** weight loader (safetensors or GGUF), BPE tokenizer (Qwen uses a byte-level BPE with ~151k vocab [INFERENCE]), RoPE, grouped-query attention (14 Q heads / 2 KV heads [INFERENCE]), KV cache and single-token incremental decode, SwiGLU MLP block, QKV bias + output projection wiring for a decoder block, tied-embedding logits, sampler (greedy/top-k/top-p), bf16/fp16 → fp32 conversion, int8/int4 dequantising matmul, fast GEMV (decode is matrix-vector).

**Memory arithmetic** (Qwen2.5-0.5B shape from memory [INFERENCE]: hidden 896, 24 layers, MLP 4864, vocab 151 936, KV heads 2 × head-dim 64, tied embeddings):

- Embedding: 151 936 × 896 = 136.1 M params.
- Per layer: Q 896² = 0.803 M; K,V 2 × 896 × 128 = 0.229 M; O 0.803 M; MLP 3 × 896 × 4864 = 13.07 M → **14.91 M**; × 24 = 357.8 M.
- Total ≈ 136.1 + 357.8 = **493.9 M params**.

| Precision | Weights | vs ESP32-S3 PSRAM 8 MB (8 388 608 B) |
|---|---|---|
| fp32 (4 B) | 493.9 M × 4 = 1 975.6 MB | 247× |
| fp16/bf16 (2 B) | 987.8 MB | 123× |
| int8 (1 B, + per-32 fp16 scales ≈ +6 %) | 493.9 MB (+30.9 MB) | 62–66× |
| int4 (0.5 B, + per-32 fp16 scales = 0.5 bit/w) | 247.0 MB (+30.9 MB) = 277.8 MB | 31–35× |

Even one layer at int8 is 14.9 MB (> 8 MB); at int4 a layer is 7.45 MB, i.e. the whole PSRAM. The int4 embedding table alone is 68 MB. KV cache per token = 24 layers × 2 (K,V) × 2 heads × 64 = 6 144 values = 12 KB at fp16; 512 tokens = 6 MB. Compute per token ≈ 2 × 357.8 M + 2 × 136.1 M (logits) ≈ 0.99 GFLOP.

Conclusion (arithmetic): Qwen 0.5B cannot be held on an ESP32-S3; streaming 248 MB of int4 weights per token from SD/flash would cost minutes per token at SD-SPI rates [INFERENCE on bandwidth]. The realistic targets are the host and the K230 Linux side. Host estimate from measured kernels (shared host, see [[W2-code-optimisation]] for noise): eager FP32 MatMul reaches 1.3 GFLOP/s at 512×512 (`BenchmarkMatMul_Eager_Large_512x512`, 209 ms/op) → ≈ 0.8 s/token single-threaded fp32 at 0.99 GFLOP/token; the gonum-backed Gorgonia path reaches 17 GFLOP/s (15.8 ms/op) → ≈ 0.06 s/token; the repo's own Exp7 loop order in plain Go reaches 12 GFLOP/s in a probe. Decode is GEMV (one vector × each weight matrix), which is memory-bound: ~2 GB of fp32 weights read per token puts the floor at memory bandwidth, so int8/int4 weights matter on the host too. A microcontroller-sized "LLM" means a ~1–5 M-param model (≤ 5 MB int8).

## (d) TinyGo face recognition

Nothing exists: no face detector, no embedding network, no pure-Go model loader (the TFLite path is CGO + linux, [[tflite]]; vision is GoCV/OpenCV, [[vision]]). Kernel coverage for a MobileFaceNet-class network [INFERENCE: ~1 M params, ~1 MB int8]: Conv2D, DepthwiseConv2D, BatchNorm, GlobalAvgPool exist in eager FP32 ([[conv-variants-depthwise-group-dilated-3d]], [[batch-norm]]); PReLU, BN folding, int8 kernels, a TinyGo-loadable weight format and a camera frame → tensor path ([[camera-io]]) are missing. 112×112×3 FP32 input = 150 KB; activations of the first stage (56×56×64 FP32) = 803 KB — fits PSRAM, not internal SRAM.

## Gaps

- No correctness gate for kernels against an external reference (numpy/PyTorch fixtures); tests are hand-computed small cases.
- No int8 (or fixed-point) kernel family; dtype tags without quantisation metadata.
- No model file format usable without CGO (weights in, weights out), apart from generic marshallers ([[marshallers]]).
- No decode-time ops (KV cache, RoPE, GQA, sampler, tokenizer).
- TinyGo build of any numeric package never attempted.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Rename `pooling_test_edgecases.go` → `_test.go` to drop testify from the production import graph — x/math/nn/layers/pooling_test_edgecases.go #p2
- [ ] Make GPTAttention/GPTEncoding forward run and add value tests against a reference — x/math/nn/layers/gpt_attention.go:280-296, gpt_encoding.go:193-218 #p1 #ER-029
- [ ] TinyGo build probe (needs TinyGo install approval) of `primitive/fp32` then `tensor/eager` for `esp32`/`esp32s3` targets — none run #p2
- [ ] float32 scalar API variant (or generic element type) to avoid soft-float on ESP32 — tensor/types/core.go #p2
- [ ] Workspace/arena contract for kernels that allocate per call (Conv2D im2col) — primitive/fp32 conv path #p2
- [ ] int8 GEMM/GEMV + per-channel scale/zero-point tensor metadata (ER-004) — tensor/types/dtype.go #p2
- [ ] Decoder-block building blocks: RoPE, GQA, KV cache, SwiGLU, sampler, safetensors/GGUF reader, BPE tokenizer (ER-012/013) — absent #p2

## Inferred objectives (proposals)

- **Proposal P1 — "TinyGo-clean core":** make `primitive/fp32` + a float32-only subset of `tensor/eager` + `nn/layers` build under TinyGo for esp32s3 with zero allocations per inference after `Init`; measure on the owner's board.
- **Proposal P2 — "LLM on host first":** implement a Qwen2-style decoder (safetensors fp16/bf16 → fp32, RoPE, GQA, KV cache, greedy decode) on host using the eager backend; compare logits against a local reference (ER-011/014). Later int8 weights; K230 Linux as second target.
- **Proposal P3 — "micro-LLM / TinyStories-class on ESP32-S3":** a 1–5 M-param model in int8 from PSRAM as the article-grade on-device demo, instead of Qwen 0.5B.
- **Proposal P4 — "Face embedding on ESP32-S3":** MobileFaceNet-class int8 network, weights exported from a pure-Go format, as the TinyGo vision demo (feeds W4/W5).

## Research questions for ideation

1. Does TinyGo's Xtensa backend (LLVM fork) support the ESP32-S3 PIE/SIMD instructions via inline asm or intrinsics, or must kernels go through ESP-DSP/ESP-NN C libraries (CGO is unavailable in TinyGo for Xtensa?)?
2. What GOPS does an ESP32-S3 reach on int8 GEMV with ESP-NN vs plain C vs TinyGo-generated code? (sets the micro-LLM tokens/s budget)
3. Which tiny language model (TinyStories 1–5 M, SmolLM-135M with aggressive quantisation, etc.) gives a meaningful demo inside 8 MB PSRAM + 16 MB flash?
4. K230: can Go on the Linux core call the KPU runtime without CGO, or is a C shim mandatory? What is the measured C908 RVV GEMM rate in Go (no RVV autovectorisation in gc)?
5. Weight format: safetensors (simple, mmap-able) vs GGUF (quantisation types built in) for a pure-Go loader that also serves TinyGo?
6. Should the framework split into a float32-only "embedded" API and the current `any`-typed general API, or move to generics (`Tensor[T]`)?
