---
type: algorithm
status: tested
tags: [easyrobot, nn, tensor, softmax]
workstream: [W9]
module: "[[tensor]]"
summary: "correct rank ≤ 2 contiguous; no rank-4 softmax (attention), no LogSoftmax"
source: [x/math/tensor/eager/activations.go, x/math/primitive/fp32/activations.go]
tests: [x/math/tensor/eager/activations_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → TestSoftmax pass (values); probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): Softmax1DGrad vs numeric 3.5e-6; rank-3 input panics"
---
# Softmax and SoftmaxGrad

**What it does:** Numerically stable softmax along one dimension (subtract the max, exp, normalise) and its vector-Jacobian product. Rank 1 and rank 2 only.

**Where:**
- Eager `x/math/tensor/eager/activations.go`: `Softmax(dim, dst)` 330-391, `SoftmaxGrad(dst, gradOutput, dim)` 269-325.
- fp32 `x/math/primitive/fp32/activations.go`: `Softmax1D(src,dst,n)` 100-133, `Softmax2DRows` 137-170, `Softmax2DCols` 174-209, `Softmax1DGrad` 213-228, `Softmax2DRowsGrad` 232-251, `Softmax2DColsGrad` 255-276.
- Plan (declared, not implemented): `x/math/tensor/eager/SOFTMAX_MULTIDIM_PLAN.md` proposes an `(outer, axis, inner)` decomposition with a scratch buffer for arbitrary rank.

**Inputs → outputs:**
- Rank 1: any dim 0.
- Rank 2:
  - `dim=0` normalises down each column; `Softmax2DRows` is a misleading name, but it is axis-0 semantics.
  - `dim=1` normalises each row; probe rows [1,2,3] gave [0.090 0.245 0.665].
- Rank ≥3 panics "only 1D and 2D tensors supported" (activations.go:384, 318). The probe reproduced the panic for `[1,2,3]`, dim 2.
- With nil dst it works in place on the receiver. Probe [1,2,1000] gave [0 0 1], so the max-subtraction keeps it stable.
- `SoftmaxGrad` uses `dx = y ⊙ (dy − Σ dy⊙y)` and takes the forward output as the receiver.
- There is no log-softmax and no masked or causal softmax. FP32 only.

## Completeness review

- Complete: no, for transformer use. Attention scores `[B,H,T,T]` (Qwen) need rank-4 softmax over the last dim, and there is no log-softmax, which cross-entropy needs for stability. The workaround is to reshape to 2D `[B·H·T, T]` and use dim=1; that works only on contiguous tensors.
- Tested: `TestSoftmax` (activations_test.go:117-184) checks 1D and 2D values. There is no test of `SoftmaxGrad`. Pass (2026-11-09).
- Correct: yes, for rank ≤2 contiguous inputs. `Softmax1DGrad` vs central differences: maxAbsErr 3.5e-6 (probe p2). The rank-2 grads use the same formula per row or column (derivation). An empty row with sum 0 is left unnormalised (`if sum > 0`).
  - The eager path ignores offset and strides. In the dst path it copies with strides, but `Softmax1D` reads `t.Data()` raw (activations.go:367-369).

## Embedded suitability

- Float64 `math.Exp` per element (fp32/activations.go:121, 157, 196) is a soft-float double on ESP32.
- No allocations in the kernels.
- `Softmax2DRows` walks columns with stride `cols`, which is cache-hostile.
- No goroutines.

## TODOs

- [ ] Implement arbitrary-rank softmax and grad per SOFTMAX_MULTIDIM_PLAN (last-dim fast path for attention) — x/math/tensor/eager/activations.go:330-391,269-325 #p2
- [ ] Add LogSoftmax (and optionally a fused causal-mask softmax for decoder attention) — activations.go #p2
- [ ] Add a numerical-gradient test for SoftmaxGrad (1D, both 2D dims) — x/math/tensor/eager/activations_test.go #p2
- [ ] Use math32.Exp instead of float64 math.Exp — x/math/primitive/fp32/activations.go:121,157,196 #p2
- [ ] Honour offset/strides in eager Softmax — activations.go:346-387 #p1 #ER-020

## Links

- Module: [[tensor]]
- Related: [[activation-kernels]], [[tensor-matmul]], [[nn]]
