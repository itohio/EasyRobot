# GoCV Tensor Backend Implementation Plan

## Objective

Design and implement a GoCV-backed tensor type that satisfies `pkg/core/math/tensor/types.Tensor` for vision-centric workloads. The backend should operate entirely within the EasyRobot repository and avoid falling back to other tensor implementations.

## Guiding Constraints

- **EasyRobot only**: All code lives under `pkg/core/math/tensor/gocv` in this repo.
- **Self-contained**: No delegation to eager, gorgonia, tflite, or external repos.
- **Image-first semantics**: Optimise for 2D images with channel dimension `[H, W, C]`, matching GoCV’s Mat layouts.
- **Explicit lifecycle**: Manage `gocv.Mat` ownership to prevent leaks/double free.
- **Surface clarity**: Unsupported tensor API methods panic with actionable errors.

## Deliverables

1. `Tensor` wrapper around `*gocv.Mat` implementing core tensor methods (`ID`, `DataType`, `Shape`, `Data`, `Clone`, `Release`, `At`, `SetAt`, etc.).
2. Operations suited to GoCV (transpose, reshape when contiguous, scalar fill, channel permutations, color conversions) implemented without external fallbacks.
3. Conversion helpers:
   - `FromMat(mat gocv.Mat) (types.Tensor, error)`
   - `ToMat(t types.Tensor) (gocv.Mat, error)` (backend-only)
   - `ToImage/FromImage` bridging `image.Image` and `store.ImageGetter`.
4. Tests verifying dtype/shape detection, clone semantics, conversions, and supported ops.
5. Package documentation outlining capabilities, limitations, and usage examples.

## Progress (2025-11-11)

- **Tensor wrapper core**: `FromMat` with ownership options, accessors (`Shape`, `Rank`, `Size`, `Empty`, `Strides`), `At`/`SetAt` for `uint8` and `float32`, `Clone`, `Copy`, `Release`, `Fill`, and limited `Reshape`/`Transpose` implemented. Unsupported interfaces panic via stubs.
- **Interface mapping**: `matTypeToDataType` / `dataTypeToMatType` added; `types.DataType` extended with pooled `UINT8`.
- **Conversions**: `FromImage`/`ToImage`, `ToMat` helpers in place (RGB/BGR conversions and store integration still pending).
- **Docs & tooling**: `doc.go`, `INSTALL_GOCV.md`, and `Makefile.cv` published for OpenCV toolchain guidance.
- **Outstanding**: channel permutation helpers, color-space utilities, store bridge, comprehensive tests, SPEC updates, and broader operation coverage.
## Work Breakdown

1. **Interface mapping**
   - Map GoCV depth/channel to `types.DataType`.
   - Add GRAY8, RGB24, RGBA, BGR types according to OpenCV matrix types.
   - Define stride/contiguity reporting.
   - Decide on ID generation and zero-value behaviour.
2. **Core wrapper**
   - Constructors that take/clone Mat ownership.
   - Implement core accessors (`Rank`, `Size`, `Empty`, `Strides`, `DataWithOffset`).
   - Implement `At`/`SetAt` for uint8 and float32 channels; panic for unsupported dtypes.
   - Ensure `Clone` performs deep copy; `Release` closes Mats exactly once.
3. **Supported operations**
   - Implement `Transpose`, `Permute` (channel swaps), `Reshape` when stride-compatible, scalar fill via `Mat.SetTo`.
   - Document unsupported math/graph operations via panics.
4. **Conversion utilities**
   - Lossless conversion Mat ↔ Tensor ↔ `image.Image`.
   - Helpers for BGR↔RGB handling and integration with `pkg/backend` store getters.
5. **Testing**
   - Table tests for depth/channel combos.
   - Round-trip conversion checks (Mat→Tensor→Mat/Image).
   - Memory safety tests (clone vs release, repeated conversions).
6. **Documentation**
   - Package-level comment or README summarising usage patterns.
   - Update relevant SPECs (vision, tensor) if necessary.

## Risks & Mitigations

- **Wide Tensor interface**: Provide clear panics for unsupported methods; unit tests ensure the implemented subset works reliably.
- **Color ordering ambiguity**: Offer explicit color conversion utilities and document defaults.
- **Resource management**: Centralise ownership rules and cover them with tests to avoid leaks.

## Success Criteria

- Vision pipelines can wrap a `gocv.Mat` as `types.Tensor`, perform supported operations, and convert back without data loss.
- Tests in `pkg/core/math/tensor/gocv` pass (`go test ./pkg/core/math/tensor/gocv`).
- No other repositories or backends are touched during the implementation.

# GoCV Tensor Backend Implementation Plan

## Objective

Introduce a GoCV-powered tensor backend that wraps `gocv.Mat` while conforming to the `pkg/core/math/tensor/types.Tensor` interface. The focus is on vision workloads so vision algorithms can use tensors without leaving GoCV.

## Guiding Constraints

- **Stay in EasyRobot**: All work happens in `pkg/core/math/tensor/gocv`.
- **No fallback backends**: This implementation must not defer to eager, gorgonia, or tflite; unsupported ops should fail loudly.
- **Image-first semantics**: Prioritise shapes `[height, width, channels]` and channel-aware operations (BGR ⇄ RGB).
- **Lifecycle safety**: Manage `gocv.Mat` ownership explicitly to avoid leaks/double frees.
- **Clear coverage**: Document which tensor methods are implemented vs. panicking.

## Deliverables

1. `Tensor` wrapper with core interface methods (`ID`, `DataType`, `Shape`, `Data`, `Clone`, `Release`, etc.).
2. Supported operations leveraging GoCV (transpose, reshape when contiguous, scalar fill via `SetTo`, color/channel utilities).
3. Conversion helpers:
   - `FromMat(mat gocv.Mat) (types.Tensor, error)`
   - `ToMat(t types.Tensor) (gocv.Mat, error)` (only for this backend)
   - `ToImage/FromImage` bridging `image.Image` and `store.ImageGetter`.
4. Tests covering dtype/shape detection, clone semantics, conversions, and representative ops.
5. Package docs summarising capabilities and limitations.

## Progress (2025-11-11)

- Core tensor wrapper with ownership semantics, accessors, `At`/`SetAt`, `Clone`, `Copy`, `Release`, and limited `Reshape`/`Transpose` is in place.
- Data type mapping extended to include pooled `UINT8`; `IsContiguous` defers to GoCV continuity.
- Conversion helpers (`FromImage`, `ToImage`, `ToMat`) implemented; RGB/BGR utilities and store glue pending.
- Package documentation (`doc.go`) plus tooling (`INSTALL_GOCV.md`, `Makefile.cv`) published.
- Tests, channel permutation utilities, color-space helpers, and SPEC updates remain outstanding.
## Work Breakdown

1. **Interface mapping**
   - Map GoCV depth/channel to `types.DataType`.
   - Decide stride/contiguity semantics for Mats.
2. **Core wrapper**
   - Build constructor(s) that take ownership of `gocv.Mat`.
   - Implement core accessors and `At/SetAt` for common dtypes (uint8, float32).
   - Ensure `Clone` deep copies data; `Release` closes Mats.
3. **Operation support**
   - Implement feasible ops (transpose, reshape, permute channels, scalar fill).
   - Panic for unsupported API subsets with guidance to convert to eager tensors.
4. **Conversions**
   - Lossless conversion to/from `image.Image` (handling color ordering).
   - Integration glue with `pkg/backend` for `store.ImageGetter`.
5. **Testing**
   - Table-driven tests for depth/channel combos.
   - Round-trip conversion checks (Mat → Tensor → Mat/Image).
   - Memory-safety tests (clone vs release).
6. **Documentation**
   - README or package doc describing usage, supported ops, limitations, and examples.

## Risks & Mitigations

- **Interface breadth**: `types.Tensor` is large; we’ll implement vision-relevant subset and document panics. Use build-tag or compile-time assertions to confirm interface conformance.
- **Color ambiguity**: GoCV defaults to BGR; provide explicit conversion helpers to avoid hidden channel swaps.
- **Resource leaks**: Wrap Mats with clear ownership semantics and test repeated conversions.

## Success Criteria

- Vision pipelines can wrap a `gocv.Mat` as `types.Tensor`, perform supported operations, and convert back without data loss.
- Tests in `pkg/core/math/tensor/gocv` pass (`go test ./pkg/core/math/tensor/gocv`).
- No other repositories are modified.


