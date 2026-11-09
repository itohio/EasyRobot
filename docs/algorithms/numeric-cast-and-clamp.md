---
type: algorithm
status: partial
tags: [easyrobot, types, conversion, tensor]
workstream: W9
module: "[[type-conversion]]"
summary: "`st.Convert` does nothing for float64/int64 destinations (`st/convert.go:25-68`); no uint8…"
source: [x/types/convert.go, x/types/convert_st.go, x/types/convert_mt.go, x/types/cast_st.go, x/types/cast_mt.go, x/types/clamp.go, x/types/clamp_i32.go, x/types/clamp_i64.go, x/types/elem_copy.go, x/types/blas.go, x/types/copy_vector.go, x/types/value.go, x/types/st/convert.go, x/types/st/convert_ops.go, x/types/st/cast.go, x/types/st/cast_i64.go, x/types/st/cast_noclamp.go, x/types/internal/constraints.go]
tests: [x/types/cast_test.go, x/types/clamp_test.go, x/types/copy_test.go, x/types/blas_test.go, x/types/value_test.go, x/types/st/cast_test.go, x/types/st/convert_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/types 27/27 pass; x/types/st build-fail on darwin/arm64 (convert_bench_test.go:583), GOARCH=amd64 go test ./x/types/st (Rosetta, 2026-11-09) ok 18/18; probe tmp/easyrobot/tensor/InfraNotes/conv"
---
# Numeric cast, clamp and strided copy (x/types)

**What it does:** Two different element-conversion families on `[]T` slices:
1. **Cast family** (`CastClamped*`, `CastClampedAny`, `CopyClamped*Any`, `ValueConvert`, `ClampTo*Value`): numeric `T(x)` with saturation to the destination integer range. Used by tensors (eager `Copy`, `CloneTensorDataTo`, conv/linalg dtype changes).
2. **Convert family** (`Convert`, `ConvertStrided`, `ConvertAny`): *audio-style normalised* conversion — float `[-1,1]` ↔ full integer range (`×127`, `×32767`, `/256` …) plus TPDF dither on down-conversion. Used by `x/marshaller/audio/sample_ring.go:363-384,553-563`. Not a numeric cast.
Plus BLAS-style `Copy/CopyStrided/CopyStridedShaped/Swap*` (`x/types/blas.go:9-145`).

**Where:**
- `x/types/st/cast.go:14-27` `CastClamped` → `elemConvertNumeric` `:31-122` (switch on `any(dst)`/`any(src)`), clamp loops `:1307-1324` (int8), `:1348-1365` (int64), `:1389-1406` (int32); strided `:598-678`.
- `x/types/clamp.go:58-104` `CastClampedAny` + a **second, duplicate** set of clamp loops `:356-513`, strided `:527-583`.
- `x/types/elem_copy.go:7-149` `CopyClampedAny`, `:155-297` `CopyClampedStridedAny` (eager `Copy`, `x/math/tensor/eager/tensor.go:323`).
- `x/types/st/convert.go:11-69` `Convert`, `:72-91` `ConvertStrided`; kernels `x/types/st/convert_ops.go:29-709`; dither `x/types/st/convert_dither.go:5-38`.
- Build selection: `convert_st.go`/`cast_st.go`/`cast_vector_st.go` (`!use_mt`) vs `*_mt.go` (`use_mt`).
- Int-size split by explicit GOARCH lists: `x/types/internal/constraints_i32.go:1` (386‖arm‖mips‖mipsle), `constraints_i64.go:1` (amd64‖arm64‖ppc64…‖s390x); same in `st/convert_i32.go`, `st/cast_i32.go`, `clamp_i32.go`, `value_i32.go` and their `_i64` twins.

**Inputs → outputs:** `dst []T, src []U, n` (n clipped to `min(len)`); strided: `shape []int`, row-major element strides (nil ⇒ contiguous, `EnsureStrides`). Supported element types: float64, float32, int64, int, int32, int16, int8, complex64. **No `uint8`, no fp16.** `*Any` variants take `any` and type-switch.

Semantics observed (probe, darwin/arm64):
| case | result |
|---|---|
| `CastClamped` f32 `[-1.7, 1.7, 127.9]`→int8 | `[-1 1 127]` — truncation toward zero, not rounding |
| `CastClamped` f32 `2^31`→int32 | `2147483647` (only because arm64 FCVTZS saturates; the guard `val > U(MaxInt32)` at `st/cast.go:1398` is false for `2^31` since `float32(MaxInt32)==2^31`) |
| `CastClamped` f32 NaN→int8 / f64 NaN→int32 | `0` / `0` (no NaN branch; result is GOARCH-defined — amd64 CVTTSS2SI yields `0x80000000` [INFERENCE]) |
| `Convert` f32→f64, i32→i64; `ConvertAny` f32→f64 | **dst untouched** (silent no-op) |
| `Convert` f32 `[0 .5 1 -1 3 .25]`→int8 | `[0 63 126 -126 127 32]` (×127 + dither, truncate) |
| `Convert` int16 `256`×4→int8 | `[0 0 1 1]` (dither) |
| `Convert` int8 `-128`→f32 | `-1.007874` (scale 1/127, asymmetric) |
| `ConvertStrided` int16→int8 non-contiguous | dst untouched (src case missing) |
| `CastClampedAny` uint8→f32 | returns `nil`, dst untouched |
| `CopyClampedStridedAny` f32→uint8 | dst untouched |
| `CopyClampedStridedAny` f32→int8 transposed `[1 2;300 -4]ᵀ` | `[1 127 2 -4]` ✔ |

## Completeness review

- Complete: **no**.
  - `st.Convert` has no `[]float64`/`[]int64` destination case (`x/types/st/convert.go:25-68`) → silent no-op; also `ConvertAny`/`convertWithScaling` route float64/int64 dst into it (`x/types/convert.go:69-74`).
  - `convertStrided` handles only dst int8/int16/int32/float32/complex64 and 1–2 source types each (`x/types/st/convert_ops.go:567-578`, `:583-609`, `:614-637`, `:642-665`, `:670-686`, `:691-707` — "Add other source types as needed"); everything else is a silent no-op.
  - No `uint8` anywhere in cast/convert (`x/types/internal/constraints.go:22-29` `NumberReal` = Float|Signed) although tensors carry `UINT8` data (`x/math/tensor/types/dtype.go:31,100`).
  - `use_mt` build is broken: `cast_mt.go:12,16` call `mt.CastClamped*` which do not exist in `x/iter/mt`; `cast_vector_mt.go:6` imports nonexistent `x/convert/mt`; `x/iter/mt` imports `x/types` → import cycle (`go build -tags use_mt ./x/types/` fails, 2026-11-09). `convert_mt.go:7` TODO.
  - wasm/any unlisted GOARCH: neither `_i32` nor `_i64` file is compiled → `x/types/st/cast.go:82` `elemConvertToInt` undefined, `st/convert.go:53` `convertToInt` undefined; `GOOS=js GOARCH=wasm go build ./x/types/...` fails (lead-reported 2026-11-09). linux/arm (GOARM=7) build ok (this run). TinyGo/Xtensa GOARCH value not checked [INFERENCE: risk].
- Tested: `x/types` 27 tests pass (`TestElemConvert*`, `TestValueConvert*`, `TestCopy*`, `TestSwap*`, `TestElem*Strided`) — table tests with expected values incl. clamping at int8/int16 bounds. `x/types/st` 18 tests (`TestElemConvertClamping`, `TestConvertFloat32ToInt8`, `TestConvertDitheringQuality`, `TestConvertStrided`, …) **do not build on darwin/arm64** because `convert_bench_test.go:583` declares an amd64-asm function without build tag; under `GOARCH=amd64` (Rosetta) they pass. No test covers float64/int64 `Convert`, uint8, NaN, `2^31` edge, or strided non-float sources.
- Correct: **no** — silent no-ops above (probe), platform-dependent NaN/overflow results, truncation instead of round-to-nearest (matters for any quantisation use).

**Defects**
- `x/types/st/convert.go:25-68` — Convert to float64/int64 is a no-op.
- `x/types/st/convert_ops.go:554-579` + per-dst kernels — strided Convert missing most source types; also uses fixed seed `0x123456789ABCDEF0` (`:592`) whereas contiguous path seeds from `src[0]` (`:60-63`) → contiguous and strided outputs differ for the same data.
- `x/types/st/convert_ops.go:80,113,146,273` — `int8(result)` truncates; with TPDF ±1 dither an exact input (int16 256) yields 0 or 1.
- `x/types/st/cast.go:1357,1398` / `x/types/clamp.go:44,33` — upper guard uses `U(MaxIntN)` which rounds up to `2^N-1+1` for float32 (and 2^63 for float64) → value exactly `2^31`/`2^63` falls into the unguarded `intN(val)` conversion (implementation-defined).
- `x/types/st/cast.go:1311-1323` (and siblings) — no NaN handling.
- `x/types/clamp.go:58-104`, `x/types/elem_copy.go:7-297` — unsupported types (uint8, complex) silently ignored; `CastClampedAny` returns nil without error; breaks `CloneTensorData([]uint8)` (see [[quantised-dtypes-and-fixed-point]]).
- `x/types/clamp.go:356-513` duplicates `x/types/st/cast.go:1286-1449` clamp loops (two implementations to keep in sync).

**Embedded suitability:** generic, allocation-free contiguous paths (`CastClamped`: one `any` type switch per call, not per element); strided paths use stack `[MAX_DIMS]int` arrays. `*Any` APIs type-switch on `any` per call. Dither path uses a heap `*xorshift64Star` per call (`convert_dither.go:14`) and float32 math per element. No float64 in hot loops except float64 inputs. No goroutines in default (`!use_mt`) build. Code size: every T×U instantiation is compiled (≈8×8 per family, three families) — relevant for TinyGo flash.

## TODOs

- [ ] Add float64/int64 (and uint8) destinations to `st.Convert`/`ConvertAny` or return an error — x/types/st/convert.go:25-68 #p1 #ER-020
- [ ] Complete `convertStrided` source/dest matrix or fall back to per-element generic path — x/types/st/convert_ops.go:554-707 #p1 #ER-020
- [ ] Replace GOARCH lists with one `strconv.IntSize`/`math.MaxInt`-based file so wasm/unknown arches build — x/types/internal/constraints_i32.go:1, x/types/st/cast_i32.go:1, x/types/st/convert_i32.go:1 #p1 #ER-026
- [ ] Fix or delete `use_mt` variants (nonexistent `mt.CastClamped`, `x/convert/mt`, import cycle) — x/types/cast_mt.go:12, x/types/cast_vector_mt.go:6 #p1 #ER-026
- [ ] Add uint8 to `NumberReal`/cast switches (image and int8-quant data) — x/types/internal/constraints.go:22, x/types/clamp.go:64 #p1 #ER-020
- [ ] Explicit NaN→0 and exact-bound handling in clamp loops (compare in float64 or use `>=` against `2^N`) — x/types/st/cast.go:1311,1357,1398 #p2
- [ ] Offer round-to-nearest-even cast for quantisation — x/types/st/cast.go:1321 #p2
- [ ] Return an error (or panic) instead of silent no-op for unsupported types in `*Any` — x/types/clamp.go:126, x/types/elem_copy.go:148 #p2
- [ ] Move `convertInt16ToFloat32Assembly` decl to an `_amd64_test.go` file so st tests build on arm64 — x/types/st/convert_bench_test.go:583 #p1 #ER-026
- [ ] Rename/document Convert as normalised audio conversion; deduplicate clamp loops in clamp.go vs st/cast.go — x/types/convert.go:7, x/types/clamp.go:356 #p3

## Links

- Module: [[type-conversion]]
- Related: [[simd-convert-amd64]], [[quantised-dtypes-and-fixed-point]], [[tensor-iterators]], [[shape-strides-iteration]]
