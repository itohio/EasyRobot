---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, normalization, llm]
workstream: [W9]
module: "[[tensor]]"
summary: "forward correct (Qwen-style, last dim, γ only); dx wrong for γ ≠ 1 and O(D²) per row"
source: [x/math/tensor/eager/normalizations.go, x/math/primitive/fp32/normalization.go]
tests: [x/math/tensor/eager/normalizations_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestRMSNormForward pass (shape only); probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): forward maxAbsErr 1.2e-7 vs float64 ref; RMSNormGrad dx maxAbsErr 6.64 for gamma≠1 (2.4e-5 for gamma=nil)"
---
# RMS normalization (forward + grad)

**What it does:** Computes `y = x / √(mean(x²) + ε) · γ` over the **last dimension**. There is no centering and no β, which is the Qwen/LLaMA `RMSNorm` form.

**Where:**
- Eager `x/math/tensor/eager/normalizations.go`: `RMSNormForward(dst, gamma, eps)` 238-264, `RMSNormGrad(dxDst, dγDst, dy, input, γ, ε)` 274-321.
- fp32 `x/math/primitive/fp32/normalization.go`: `RMSNormForward` 137-175, `RMSNormGrad` 551-599.

**Inputs → outputs:**
- x `[..., D]`; γ `[D]` or nil.
- **Normalises over the last dim only and has γ only (no β).** ε is inside the square root, mean over D. This matches HF Qwen2 `Qwen2RMSNorm` (`x * rsqrt(mean(x²)+eps) * weight`, eps 1e-6) [INFERENCE: HF formula from memory, not checked here].
  - HF computes the variance in float32 after upcasting from bf16; here everything is float32.
- dγ **accumulates** (`+=`).
- FP32 only. Contiguous buffers.

## Completeness review

- Complete: yes for the LLM use-case forward.
- Tested: `TestRMSNormForward` (normalizations_test.go:106-133) checks **shape only**. No grad test. Pass (2026-11-09).
- Correct: **no** (gradient).
  - Forward: correct. Probe `[2,4]`, γ=[0.5,1,2,-1], ε=1e-6: maxAbsErr 1.2e-7 vs a float64 reference.
  - `RMSNormGrad`: the correction term uses `Σⱼ dyⱼ·xⱼ`; it needs `Σⱼ γⱼ·dyⱼ·xⱼ` (normalization.go:589-595), so dx is wrong whenever γ≠1. Probe: γ≠1 gives maxAbsErr 6.64 (|dx|≈2.4); γ=nil gives 2.4e-5.
  - That inner sum is also recomputed for every i, making the gradient **O(D²)** per row (normalization.go:592-595). For Qwen-0.5B, D=896 means ~800 k MACs per row instead of ~2.7 k.
  - Eager forward with nil dst aliases the input via `NewAs` (normalizations.go:242-244).
  - No caller outside eager/gorgonia: no `nn` layer uses RMSNorm (grep, 2026-11-09).

## Embedded suitability

No allocation in the kernel. It divides per element instead of multiplying by a hoisted reciprocal (normalization.go:165). float32 `math32`. Good fit for MCU inference once fused with the γ multiply; it already is fused.

## TODOs

- [ ] Fix RMSNormGrad dx for γ≠1 and hoist the Σ(γ·dy·x) out of the per-element loop (O(D²)→O(D)) — x/math/primitive/fp32/normalization.go:579-596; probe maxAbsErr 6.64 #p1 #ER-020
- [ ] NewAs aliasing: allocate a dst when nil — x/math/tensor/eager/normalizations.go:242-244 #p1 #ER-020
- [ ] Add forward value tests (vs a Qwen2RMSNorm reference) and a numerical grad test — x/math/tensor/eager/normalizations_test.go:106-133 #p2
- [ ] Add an nn RMSNorm layer for the Qwen path — no caller in nn/ #p2
- [ ] Multiply by a precomputed 1/rms instead of dividing — normalization.go:165 #p3

## Links

- Module: [[tensor]]
- Related: [[layer-norm]], [[nn]], [[tensor-matmul]], [[softmax-kernel]]
