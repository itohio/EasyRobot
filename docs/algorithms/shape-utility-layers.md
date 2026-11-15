---
type: algorithm
status: partial
tags: [easyrobot, nn, layers]
workstream: W9
module: "[[nn]]"
summary: "`Squeeze`/`Unsqueeze` `Backward` panic (`utility.go:427,587`); `PadReflect` zero-pads…"
source: [x/math/nn/layers/utility.go, x/math/nn/layers/activations.go]
tests: [x/math/nn/layers/utility_test.go, x/math/nn/layers/activations_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestFlatten, TestReshape, TestSqueeze, TestUnsqueeze, TestTranspose, TestDropout(+_EdgeCases) pass; probes (2026-11-09): Squeeze/Unsqueeze Backward panic, PadReflect wrong"
---
# Shape and utility layers: Flatten, Reshape, Squeeze, Unsqueeze, Transpose, Pad, Concatenate, Dropout

**What it does:** parameter-free layers that move or copy data between shapes, plus Dropout regularisation.

**Where:** `x/math/nn/layers/utility.go`
- `Flatten` :12-162 (`NewFlatten(start,endExclusive)`)
- `Reshape` :165-315
- `Unsqueeze` :318-449
- `Squeeze` :452-609
- `Transpose` :612-744 (2D only)
- `Pad` / `NewPadReflect` :746-975
- `Concatenate` :978-1325

`Dropout` lives in `activations.go:393-574`.

**Inputs → outputs:** any shape. Forward copies into a preallocated output (`Reshape` + `Copy`). Backward copies or reshapes back. Kernels in [[tensor]]: `Reshape`, `Copy`, `Transpose`, `DropoutMask`, `DropoutForward`, `Multiply`.

## Completeness review

- Complete: no.
  - `Transpose` rejects anything but 2D (:645, :667). `NewTransposeDims` accepts dims that are then ignored.
  - `Pad` reflect mode is accepted (`NewPadReflect`, :771-781), but `Forward` always does a constant fill and ignores `mode` (:815-842).
  - `Concatenate` does not implement `types.Layer`: `OutputShape([][]int)` (:1293). It cannot be added to `SequentialModelBuilder`. Its `Backward` returns only input 0's gradient (:1196-1202).
  - Dropout has no global train/eval switch: each layer carries its own `isTraining`, default **false** (:439).
- Tested:
  - `TestFlatten`, `TestReshape`, `TestSqueeze`, `TestUnsqueeze`: shape plus data preserved, **forward only**.
  - `TestTranspose`: forward values plus backward shape.
  - `TestDropout`: inference passthrough, p=0, dropped=0, kept scaled by 1/(1−p), and backward mask applied (activations_test.go:350-525).
  - No Pad or Concatenate tests.
  - All pass.
- Correct: no.
  - Probe `utilcheck`:
    - **Unsqueeze(0).Backward PANIC** `tensor.Copy: shape mismatch: dst [2 3] vs src [1 2 3]`
    - **Squeeze().Backward PANIC** `dst [1 2 3] vs src [2 3]`
    - Reshape and Flatten backward ✓
    - Concatenate(dim 1 and dim 0) ForwardMulti/BackwardMulti values ✓
  - Probe `gradcheck`:
    - Flatten(1,4) dX maxAbs 9.4e-7 ✓
    - Pad const dX 7.4e-7 ✓
    - **PadReflect `[1 2 3]` pad(1,1) → `[0 1 2 3 0]`** (reflect expects `[2 1 2 3 2]`)

## Defects

- Squeeze/Unsqueeze Backward: `gradInput.Copy(gradOutput)` with differing shapes. It needs `Reshape` first (Unsqueeze :427, Squeeze :587). Any model using them cannot train. #p1
- Pad ignores `mode`, so reflect padding silently becomes zero padding (:815-842). #p1
- Pad copies element by element through recursive `At`/`SetAt` with per-element index-slice allocations (`computeIndices` :941-949, `make` :874, :924). This is O(N) heap allocations per call. #p2
- Concatenate is not a `types.Layer` and Backward is lossy (:1183-1203, :1293). #p2
- Flatten/Squeeze/Unsqueeze/Transpose/Pad Backward allocate `gradInput` every call (:117, :421, :581, :705, :906). #p3
- Flatten doc says "flattens to 2D" (:11), but it actually merges `[startDim,endDim)` with an exclusive end.
- Dropout uses its own `*rand.Rand` (:440) separate from `Base.rng`. `NewDropout(name, …DropoutOption)` uses a different option type from Base `Option`.

## TODOs

- [ ] Reshape gradOutput to input shape in Squeeze/Unsqueeze Backward + add backward tests — utility.go:427,587 #p1 #ER-029
- [ ] Implement reflect (and replicate) modes in Pad.Forward/Backward or remove NewPadReflect — utility.go:771-842 #p1 #ER-029
- [ ] Vectorise Pad (row copies) instead of per-element recursion — utility.go:845-938 #p2
- [ ] Make Concatenate usable in models (multi-input graph or Layer adapter) — utility.go:978-1325 #p2
- [ ] N-D Transpose using the dims permutation — utility.go:635-744 #p3
- [ ] Model-wide train/eval switch (Dropout, BatchNorm) — activations.go:561-569 #p2

## Embedded suitability

Forward copies (no views), so memory doubles per reshape layer. Backward allocates. Pad is very slow (per-element interface calls plus allocations). Dropout uses math/rand per element. No goroutines.

## Links

- Module: [[nn]] · kernels: [[tensor]] · related: [[activation-layers]], [[batchnorm2d-layer]], [[sequential-model-and-backprop]]
