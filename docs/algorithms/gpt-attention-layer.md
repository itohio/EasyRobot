---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, transformer, llm]
workstream: [W9, W2]
module: "[[nn]]"
summary: "`Forward` always panics; `Backward` zeroes gradient; no RoPE/GQA/KV cache"
source: [x/math/nn/layers/gpt_attention.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no GPTAttention test (go test -count=1 ./x/... 2026-11-09); probe tmp/easyrobot/tensor/layernotes/gradcheck (2026-11-09): Forward panics for [3,4], [1,3,4], [2,3,4]; Backward zeroes gradOutput"
---
# GPT multi-head causal self-attention layer

**What it does (declared):** fused-QKV multi-head scaled dot-product self-attention with an additive causal mask and an output projection. **In practice Forward always panics and Backward is a destructive placeholder.**

**Where:** `x/math/nn/layers/gpt_attention.go`: struct :15-30, `NewGPTAttention` :37-100 (causal mask :86-97), `Init` :104-167, `Forward` :172-209, `computeAttention` :212-248, `projectQKV` :251-269, `splitAndReshapeQKV` :272-300, `computeScaledDotProductAttention` :303-364, `concatenateAndProject` :367-390, `Backward` :394-424, `OutputShape` :428-449.

**Inputs → outputs:** `[T,D]` or `[B,T,D]` → same shape; `T ≤ maxSeqLen`.
- Params: `W_qkv [D,3D]` stored at `ParamWeights`; `W_o [D,D]` stored at **`ParamBiases`** (:80-84). Neither projection has a bias.
- Mask: dense `[maxSeqLen,maxSeqLen]` with 0 / −MaxFloat32 (:88-97).
- Kernels used ([[tensor]]): `MatMul`, `Slice`, `Reshape`, `MatMulTransposed`, `ScalarMul`, `Add`, `Softmax`, `Transpose`.

## Completeness review

- Complete: no.
  - Dropout is accepted and validated but is a TODO no-op (:327-329, single-sequence path only).
  - Backward is a placeholder: no gradients for W_qkv/W_o and no real dX (:408-423).
  - No KV cache, no incremental decode, no padding mask, no RoPE, no GQA.
- Tested: no tests in layers or learn.
- Correct: **no**. Probe `gradcheck`:
  - Forward `[3,4]`, `[1,3,4]` and `[2,3,4]` (D=4, H=2) all panic with `tensor.Slice: start+length (8+12=20) exceeds dimension size 12`.
  - Backward with gradOutput=[0.341 −0.679 −0.530 …] leaves gradOutput = [0 0 0 …] and returns gradInput = zeros.

## Defects (all #p1 unless noted)

- `Slice(dst, dim, start, LENGTH)` (eager/tensor.go:592) is called as if the last argument were an end index: `Slice(nil,1,embedDim,2*embedDim)` and `Slice(nil,1,2*embedDim,3*embedDim)` (:280-281, :290-291). K gets width 2D and V overruns, so Forward panics.
- Even with correct lengths, the head split is wrong. Reshaping a `[T,D]` strided view straight to `[H,T,hd]` (:284-286, :294-296) reinterprets memory. It needs `[T,H,hd]` followed by a transpose to `[H,T,hd]`.
- The single-sequence `attnOutput` is allocated 2D `[T,D]` (:163), but SDPA slices it as `[H,T,hd]` (:332). `concatenateAndProject` checks `len(attnOutput.Shape())==3` (:370), so it falls into the batch branch with a 4-axis `Transpose` on a 2D tensor.
- Per-head `Slice`/`Reshape` views feed element-wise ops (`ScalarMul`, `Add`, `Softmax`) that ignore view offsets (see [[lstm-layer]] root cause, eager/tensor.go:103-111), so per-head results would also be wrong.
- `concatenateAndProject` is passed `q` but never uses it (:243, :367).
- `Backward`: `gradOutput.Copy(gradInput)` copies the zero buffer **into gradOutput**, because `Copy(src)` copies src into the receiver (eager/tensor.go:300). It returns zeros and silently kills the gradient flow for everything upstream (:420).
- `projectQKV` replaces `a.qkvBuffer` with a reshaped view on every batch call (:261-265). This allocates views.
- W_o is stored as `ParamBiases` (:81-84), which breaks the parameter-index convention, and Sequential.Parameters merges it by index (see [[sequential-model-and-backprop]]). #p2
- The mask is O(maxSeqLen²) memory, e.g. 32k ctx gives 4 GiB FP32 (:88). #p2

## Qwen2-style decoder readiness

| Needed | Status |
|---|---|
| Causal mask | present (dense additive, :86-97); unusable because Forward panics |
| RoPE (rotary on q/k, θ=1e6) | **missing** (no rotary code anywhere in x/math) |
| GQA (n_kv_heads < n_heads, K/V repeat) | **missing**: single fused W_qkv `[D,3D]` assumes equal heads |
| QKV bias (Qwen2 has q/k/v bias) | **missing** |
| KV cache + single-token decode step | **missing**: Init fixes T; Forward recomputes full T×T |
| RMSNorm | no layer; tensor op exists (`RMSNormForward/Grad`, eager/normalizations.go:238,274; fp32/normalization.go:137) |
| SwiGLU MLP | no layer; `Swish` tensor op exists (eager/activations.go:664); gate/up/down = 3 Dense (rank-2 only) |
| Tied embeddings / LM head | [[gpt-token-embedding-layer]] table; no tied-head helper |
| FP16/BF16/int8 weights | layers are FP32 (Base dtype only); see [[type-conversion]] |
| Residual add / block container | missing (Sequential is linear-only, no skip connections) |

Verdict: not usable as-is. A Qwen2 forward path needs a new inference-only attention, written against contiguous buffers, with RoPE, GQA, KV cache and fused softmax.

## TODOs

- [ ] Fix QKV split lengths and head reshape/transpose; fix 2D attnOutput shape — gpt_attention.go:163,279-296,370-386 #p1 #ER-029
- [ ] Make element-wise ops honour Slice view offsets (or copy heads to contiguous scratch) — eager/tensor.go:103-111 #p1 #ER-029
- [ ] Replace placeholder Backward (reversed Copy zeroes gradOutput) with a real backward or return an explicit "not implemented" error — gpt_attention.go:408-423 #p1 #ER-029
- [ ] Add value tests vs a reference attention (tiny D, H=2, causal) — (no test) #p1 #ER-029
- [ ] Implement dropout or remove the parameter — gpt_attention.go:327-329 #p3
- [ ] Add RoPE, GQA, QKV bias and a KV-cache decode API for Qwen2 — gpt_attention.go (absent) #p2
- [ ] Store W_o under its own ParamIndex — gpt_attention.go:80-84 #p2

## Embedded suitability

O(T²) score buffer and O(maxSeqLen²) mask. Every head allocates several `Slice`/`Reshape` view headers. `math.Sqrt` in float64 runs once per call. No goroutines. Not suitable for ESP32 at any useful T. On K230 or desktop it needs a rewrite with fused kernels.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[primitive-kernels]], [[mat-vec]] · related: [[gpt-token-embedding-layer]], [[gpt-positional-encoding-layer]], [[batchnorm2d-layer]], [[activation-layers]], [[lstm-layer]]
