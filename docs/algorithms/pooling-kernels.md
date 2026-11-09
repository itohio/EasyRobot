---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, pooling]
workstream: [W9]
module: "[[tensor]]"
summary: "AdaptiveAvgPool uses floor end (≠ PyTorch); MaxPool backward splits ties; −1e30 init"
source: [x/math/tensor/eager/conv.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/tensor/eager/conv_test.go, x/math/tensor/eager/pooling_test.go, x/math/primitive/fp32/pooling_test.go, x/math/primitive/fp32/tensor_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → pooling tests pass; probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): AdaptiveAvgPool2D wrong for non-divisible/upsampling sizes; MaxPool backward splits ties"
---
# Pooling: MaxPool2D (+indices, backward), AvgPool2D (+backward), GlobalAvgPool2D, AdaptiveAvgPool2D, ScatterAdd

**What it does:** NCHW 2D pooling forward and backward, plus `ScatterAdd`, which routes gradients through linear int32 indices.

**Where:**
- Eager `x/math/tensor/eager/conv.go`: `MaxPool2D` 807-873, `MaxPool2DWithIndices` 880-963, `MaxPool2DBackward` 970-1051, `AvgPool2D` 1056-1111, `AvgPool2DBackward` 1117-1190, `GlobalAvgPool2D` 1195-1239, `AdaptiveAvgPool2D` 1623-1679, `ScatterAdd` 1868-1922.
- Primitives `x/math/primitive/fp32/tensor.go`: `MaxPool2D` 474-525, `MaxPool2DWithIndices` 534-592, `MaxPool2DBackward` 606-698, `AvgPool2D` 705-758, `AvgPool2DBackward` 768-833, `GlobalAvgPool2D` 838-869, `AdaptiveAvgPool2D` 875-923, `ScatterAdd` 1638-1676.
- Also primitive-only, no eager wrapper: `MaxPool1D`/`3D`, `AvgPool1D`/`3D`, `GlobalMaxPool1/2/3D`, `AdaptiveMaxPool1/2/3D` (tensor.go:929-1420).

**Inputs → outputs:**
- `[B,C,H,W]` with `kernelSize/stride/padding [2]` → `[B,C,OH,OW]`, where `OH=(H+2p−k)/s+1`.
- Indices are an INT32 tensor of linear indices into the input.
- `GlobalAvgPool2D` gives `[B,C]`. `AdaptiveAvgPool2D(dst, [OH,OW])`.
- `MaxPool2DBackward(dst, dy, indices, …)` needs the receiver to be the forward input.
- FP32 only.

## Completeness review

- Complete: yes for 2D. Not supported: ceil_mode, dilation, and `count_include_pad` choice. `AvgPool2D` does not check that kernel/stride/padding have length 2 (conv.go:1072-1077), so it panics with an index error.
- Tested:
  - Forward values: eager `TestMaxPool2D`, `TestAvgPool2D`, `TestGlobalAvgPool2D`, `TestAdaptiveAvgPool2D` (divisible sizes only); fp32 `TestMaxPool2DForward`, `TestAvgPool2DForwardWithPadding`.
  - Backward: `TestMaxPool2DBackward`, `TestAvgPool2DBackwardWithPadding`, `Test*ClearsDestination`, `TestTensor*BackwardMatchesPrimitive`.
  - `TestScatterAddClearsDestination`.
  - All pass (2026-11-09).
- Correct: **no** for some sizes and semantics:
  - **AdaptiveAvgPool2D** uses `end=floor((i+1)·H/out)` (tensor.go:897-900); PyTorch uses `ceil`.
    - Probe W=5→3 gave [1 2.5 4.5]; torch gives [1.5 3 4.5].
    - W=2→3 gave [0 1 2]; torch gives [1 1.5 2]. An empty window yields 0.
  - **MaxPool2DBackward** splits the gradient equally among all window elements within 1e-6 of the max (tensor.go:651-693). Probe with all-equal input gave [0.25 0.25 0.25 0.25]; PyTorch/TF route it to the argmax index only, [1 0 0 0]. The absolute 1e-6 epsilon is also scale-dependent, and the stored `indices` are used only to read the max value.
  - **MaxPool** initialises to `-1e30` rather than −Inf (tensor.go:501, 568). Probe with all inputs < −1e30 returned −1e30. A window that lies entirely in padding returns −1e30, and that window's index is −1.
  - **AvgPool** divides by the count of valid (non-pad) elements, i.e. `count_include_pad=False` (tensor.go:744-753). Probe corner: 1 vs 0.25 for torch's default `count_include_pad=True`. This is a semantics choice, but it is undocumented.
  - Eager `ScatterAdd` zeroes the **entire backing slice** of dst, including data outside a view (conv.go:1903-1906). That contradicts the primitive's "not cleared" contract (tensor.go:1630-1632).
  - View offsets are ignored, as in all eager ops.

## Embedded suitability

`MaxPool2DWithIndices` with nil indicesDst allocates `[]int32` (conv.go:938-940). Outputs are allocated when dst is nil. The kernels do not allocate; they are float32 only. The backward pass re-scans each window twice.

## TODOs

- [ ] AdaptiveAvgPool2D: use start=floor(i·H/out), end=ceil((i+1)·H/out) — x/math/primitive/fp32/tensor.go:897-900; probe [1 2.5 4.5] vs [1.5 3 4.5] #p1 #ER-020
- [ ] MaxPool2DBackward: route the gradient to the stored argmax index only (or document the tie-splitting) — tensor.go:651-693 #p2
- [ ] Initialise MaxPool with -Inf (math32.Inf(-1)) — tensor.go:501,568 #p3
- [ ] Document or parametrise count_include_pad in AvgPool2D/Backward — tensor.go:744-753,792-823 #p3
- [ ] Eager ScatterAdd: clear only the dst logical region (or not at all, per the primitive contract) — x/math/tensor/eager/conv.go:1903-1906 #p2
- [ ] Validate kernel/stride/padding lengths in AvgPool2D — conv.go:1072-1077 #p3
- [ ] Add AdaptiveAvgPool2D tests with non-divisible and upsampling sizes — x/math/tensor/eager/conv_test.go:476 #p2

## Links

- Module: [[tensor]]
- Related: [[conv2d-kernel]], [[nn]]
