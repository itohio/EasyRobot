---
type: algorithm
status: stub
tags: [easyrobot, tensor, quantisation, int8, dtype]
workstream: W9
module: "[[primitive-kernels]]"
summary: "qi*/qu8/bytes are pool declarations only; no scale/zero-point; `CloneTensorDataTo` returns zeros…"
source: [x/math/tensor/types/dtype.go, x/math/primitive/qi/int.go, x/math/primitive/qi8/int8.go, x/math/primitive/qi16/int16.go, x/math/primitive/qi32/int32.go, x/math/primitive/qi64/int64.go, x/math/primitive/qu8/uint8.go, x/math/primitive/bytes/bytes.go]
tests: [x/math/tensor/types/dtype_test.go, x/math/tensor/types/tensor_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/math/tensor/types 8/8 pass; qi*/qu8/bytes 'no test files'; probe InfraNotes/conv: CloneTensorData([]uint8{7,8}) → [0 0], CloneTensorDataTo(FP16,…) → nil"
---
# Quantised dtypes and fixed point (tensor DataType + qi*/qu8 packages)

**What it does:** Declares tensor element-type tags including low-precision ones and provides typed buffer pools. There is **no quantisation scheme**: no scale, zero-point, per-channel params, requantisation, int8 GEMM/conv kernels or fp16 storage.

**Where:**
- `x/math/tensor/types/dtype.go:16-32` `DataType` tags: `INT64, C128, C64, FP64, INT32, FP32, INT, INT16, FP16, INT8, INT48 ("4-bit unpacked into 8bit"), UINT8`.
- `dtype.go:39-78` `TypeFromData` (any→tag), `:80-105` `MakeTensorData` (pool `Get`), `:107-131` `ReleaseTensorData`, `:133-167` `CloneTensorDataTo` (alloc + `types.CastClampedAny`), `:169-174` `CloneTensorData`.
- Primitive "quantised" packages are 7-line files, each only `var Pool = buffer.Pool[T]{}`: `qi/int.go` (int), `qi8/int8.go`, `qi16/int16.go`, `qi32/int32.go`, `qi64/int64.go`, `qu8/uint8.go`, `bytes/bytes.go` (declares `package qi8`, `buffer.Pool[byte]`). Pool = tiered `sync.Pool` (`x/buffer/pool.go:15-170`).

**Inputs → outputs:** `MakeTensorData(dt, n) any` → `[]T` from pool; `CloneTensorDataTo(dt, data any) any` → new slice cast with clamping/truncation (see [[numeric-cast-and-clamp]]).

| tag | storage | MakeTensorData | cast support |
|---|---|---|---|
| INT8 | []int8 | qi8.Pool | clamp+truncate |
| INT48 | []int8 (same as INT8) | qi8.Pool | no packing, no 4-bit range clamp |
| UINT8 | []uint8 | qu8.Pool | **none** (CastClampedAny ignores uint8) |
| FP16 | — | **nil** | none |
| C128 | — | **nil** | none |
| FP64 | []float64 | `make` (unpooled) | yes; `ReleaseTensorData` ignores float64 |

## Completeness review

- Complete: **no** — tags exist without kernels/params. `FP16`/`C128` → `MakeTensorData` returns nil (`dtype.go:102-103`); `INT48` is just INT8 storage (`:98-99`); `TypeFromData` has no scalar `uint8` case (`:39-56`).
- Tested: `TestMakeTensorData`, `TestCloneTensorData(To)`, `TestTypeFromData`, `TestReleaseTensorData`, `TestGetTensorData`, `TestMust`, `TestCopyTensorData` pass; `dtype_test.go:157` checks INT48 → []int8; `:645` float32→int8 clone. No test exercises UINT8 clone/copy or FP16. The `x/math/learn` `TestQuantizeModel` is skipped (owned by [[learn]], see learn-post-training-quantization note there).
- Correct: **no** — `CloneTensorData([]uint8{7,8})` returns `[0 0]` (probe): `CastClampedAny` has no uint8 case so the fresh pool buffer is returned uncopied (`dtype.go:165`, `x/types/clamp.go:58-104`). Pool buffers are not zeroed on reuse [INFERENCE from tiered sync.Pool] → may even return stale data.

**Defects**
- `x/math/tensor/types/dtype.go:165` — clone of UINT8 (or any unsupported pair) silently yields uninitialised/zero data.
- `x/math/tensor/types/dtype.go:127-129` — comment "no pooling for uint8" contradicts the `qu8.Pool.Put` just above.
- `x/math/primitive/bytes/bytes.go:1` — package name `qi8` in directory `bytes` (import name collision/confusion).
- `x/math/tensor/types/dtype.go:30` — INT48 described as 4-bit but no clamp to [-8,7] and no packing.

**What a real int8 scheme would need (proposal):** per-tensor/per-channel `scale float32` + `zeroPoint int32` carried on the tensor (TFLite-compatible affine `real = scale·(q−zp)`); quantise = round-half-even + saturate (current cast truncates); int8×int8→int32 GEMM/conv/depthwise kernels with int32 bias and fixed-point requantisation (multiplier + shift, no float in the loop — matters on ESP32 without FPU for doubles; ESP32-S3 has PIE SIMD for int8); per-op calibration (min/max or histogram) from [[learn]]; fp16 needs a software half↔float converter (no native type in Go/TinyGo).

**Embedded suitability:** pools use `sync.Pool` + `sync.RWMutex` (TinyGo supports both; `sync.Pool` in TinyGo does not actually pool [INFERENCE]); `any`-typed data with type switches; no hot loops here.

## TODOs

- [ ] Support UINT8 in `CloneTensorDataTo` (or error) — x/math/tensor/types/dtype.go:165 #p1 #ER-020
- [ ] Return error/panic for FP16, C128 in `MakeTensorData` instead of nil, or remove tags — x/math/tensor/types/dtype.go:102 #p2
- [ ] Design quantisation params (scale/zero-point) on tensor + int8 kernels for ESP32/K230 — proposal, W9 #p2
- [ ] Fix `package qi8` in x/math/primitive/bytes — x/math/primitive/bytes/bytes.go:1 #p3
- [ ] Add UINT8/INT48 clone/copy tests — x/math/tensor/types/dtype_test.go #p2

## Links

- Module: [[primitive-kernels]]
- Related: [[numeric-cast-and-clamp]], [[tflite-model-import]], [[learn]]
