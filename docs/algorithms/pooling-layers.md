---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, pooling]
workstream: W9
module: "[[nn]]"
summary: "Max/Avg gradients correct; `GlobalAvgPool2D.Backward` returns zeros of shape `[B,C]`…"
source: [x/math/nn/layers/pooling.go, x/math/nn/layers/pooling_test_edgecases.go]
tests: [x/math/nn/layers/pooling_test.go, x/math/nn/layers/pooling_grad_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestMaxPool2D, TestAvgPool2D, TestGlobalAvgPool2D, Test{Avg,Max}PoolLayerBackwardMatchesExpected pass; probe: Max/Avg backward ✓, GlobalAvgPool2D backward wrong shape"
---
# Pooling layers: MaxPool2D, AvgPool2D, GlobalAvgPool2D

**What it does:** spatial downsampling layers for NCHW feature maps.

**Where:** `x/math/nn/layers/pooling.go`
- `MaxPool2D` :12-178: `Forward` :96-123 uses `MaxPool2DWithIndices`; `Backward` :127-157 uses `MaxPool2DBackward` with the stored indices.
- `AvgPool2D` :181-338: `Forward` :264-288 uses `AvgPool2D`; `Backward` :291-317 uses `AvgPool2DBackward`.
- `GlobalAvgPool2D` :341-448: `Forward` :375-399 uses `GlobalAvgPool2D`; `Backward` :402-432 is **not implemented**.

**Inputs → outputs:** `[B,C,H,W]` → `[B,C,(H+2p-k)/s+1,…]` for Max/Avg, `[B,C]` for Global. No parameters. Kernels in [[tensor]].

## Completeness review

- Complete: no. `GlobalAvgPool2D.Backward` returns the error "backward pass not yet implemented" when `CanLearn()` (:421-424). Otherwise (the default, since `canLearn=false`, base.go:44) it returns **zeros of shape `[B,C]`** (:428), but the input was `[B,C,H,W]`. Wrong shape and zero gradient are returned silently, and Sequential then feeds that into the previous layer. layers/SPEC.md:281 declares it "Fully implemented including backward pass".
- Tested: `TestMaxPool2D`, `TestAvgPool2D` and `TestGlobalAvgPool2D` check forward values on 4×4 / 3×3 grids (pooling_test.go:12-90). `TestAvgPoolLayerBackwardMatchesExpected` and `TestMaxPoolLayerBackwardMatchesExpected` check gradient values through Conv(1×1)→Pool (pooling_grad_test.go:13-175). All pass.
- Correct: Max/Avg yes, Global no. Probe `gradcheck`:
  - MaxPool 2×2 s2 on [1,2,4,4]: dX maxAbs 1.2e-5
  - AvgPool 3×3 s2 p1 on [1,1,5,5]: maxAbs 5e-7 (backward is consistent with the forward's padding convention)
  - AvgPool 2×2 s2 on [2,2,4,4]: 2.3e-7
  - GlobalAvgPool: gradInput shape `[1 2]` ≠ input `[1 2 3 3]`

## Defects

- `GlobalAvgPool2D.Backward` is missing and silently returns a wrong-shape zero gradient (pooling.go:421-431). Correct formula: `dX[b,c,h,w] = g[b,c]/(H·W)`.
- `pooling_test_edgecases.go` is a **non-`_test.go` file in package `layers`**. It defines exported `TestMaxPool2D_EdgeCases`, `TestAvgPool2D_EdgeCases` and `TestGlobalAvgPool2D_EdgeCases` (:12, :58, :104), imports `testing` and testify into the production library, and is **never run** by `go test`. It is absent from the 2026-11-09 JSON.
- `MaxPool2D.Forward` allocates a new indices tensor every call (`MaxPool2DWithIndices(output, nil, …)`, :115).
- Backward allocates `gradInput` on every call, because `Base.Grad()` is never preallocated by `Init` (:146-153, :306-313).
- Options types `MaxPool2DOption`/`AvgPool2DOption` (:24, :192) are unused by the constructors. The constructors take no options, so pooling layers cannot be named via `WithName`.

## TODOs

- [ ] Implement GlobalAvgPool2D.Backward (broadcast g/(H·W)) and remove the CanLearn branch — pooling.go:421-431 #p1 #ER-029
- [ ] Rename pooling_test_edgecases.go → pooling_edgecases_test.go so it runs and stops leaking `testing` into the library — x/math/nn/layers/pooling_test_edgecases.go:1-143 #p3
- [ ] Preallocate indices and gradInput in Init — pooling.go:115,146-153,306-313 #p2
- [ ] Accept Base Options (name) in pooling constructors or delete the unused option types — pooling.go:24-31,192-199 #p3

## Embedded suitability

Forward is allocation-free except for MaxPool indices. Backward allocates. FP32 kernels live in [[primitive-kernels]]. No goroutines, no float64.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[primitive-kernels]] · related: [[conv2d-layer]], [[sequential-model-and-backprop]]
