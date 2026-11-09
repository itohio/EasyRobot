---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, types, conversion, quantisation]
workstream: [W9, W2]
packages: [x/types, x/types/st, x/types/internal]
test_result: build-fail
---
# type-conversion

**Purpose:** generic numeric conversion between slices of different element types — saturating casts (`Cast*Clamped`), normalised conversion (`Convert`, audio-style scaling between int and float), strided/matrix/vector copies — used by eager `Tensor.Copy` (`CopyClampedStridedAny`), dtype casts in [[tensor]], audio sample conversion ([[audio-io]]) and marshallers.

**Packages (non-test LOC):**

| Package | Files | LOC | Notes |
|---|---|---|---|
| `x/types` | 18 | 2840 | public generic + `*Any` (type-switch) entry points: `Cast`, `CastClamped(Any/Strided)`, `Convert(Any/Strided)`, `Copy*`, `Clamp*Value`, `Swap`, BLAS-like helpers `blas.go`; `_i32`/`_i64` int-size files; `*_mt.go` (`use_mt`) |
| `x/types/st` | 10 + 1 `.s` | 3859 + 107 asm | single-threaded kernels `cast.go` (1625 LOC), `convert_ops.go`, `convert_dither.go`, `convert_asm_amd64.s` |
| `x/types/internal` | 3 | 69 | numeric constraints, int-size variants |

Declared docs: `st/CONVERT_PLAN.md`, `st/dispatch_overhead_analysis.md`, `st/dispatch_optimization_issue.md`.

## Public API

- Typed generics: `CastClamped[T,U Number](dst []T, src []U, n)`, `Convert[T,U](dst, src, n)`, `CopyStridedShaped[T](dst, src, shape, stridesDst, stridesSrc)` …
- Dynamic: `CastClampedAny(dst, src any)`, `ConvertAny`, `CopyClampedStridedAny(dst, src any, shape, dstStrides, srcStrides)` — type switches over the supported slice types; unsupported pairs are a **silent no-op**.
- Composition: tensors store `any` data, so eager calls the `*Any` forms; each call pays a two-level type switch (analysed in `dispatch_overhead_analysis.md`).

## Algorithms

<!-- vault:algorithms h=e32b621a1374 -->

Counts: 0 tested · 0 implemented · 1 partial · 1 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[numeric-cast-and-clamp]] | partial | pass | `st.Convert` does nothing for float64/int64 destinations (`st/convert.go:25-68`); no uint8… |
| [[simd-convert-amd64]] | stub | build-fail | asm writes the constant 3.05e-5 into every element (checked under Rosetta); declared only in a… |

<!-- /vault:algorithms -->

## Build targets and dependencies

- No CGO. Int-size selection by **explicit GOARCH lists** (`_i32`: 386, arm, mips, mipsle; `_i64`: amd64, arm64, ppc64(le), mips64(le), riscv64, s390x): `GOOS=js GOARCH=wasm go build ./x/types/...` fails (`st/cast.go:82: undefined: elemConvertToInt`), so the whole tensor/nn stack fails on wasm and on any TinyGo target that reports an unlisted GOARCH.
- `use_mt` variants reference nonexistent packages/functions (`cast_mt.go:12`, `cast_vector_mt.go:6`).
- Heavy generics + type switches (175 switches, 283 `float64` tokens in non-test code) → code size on TinyGo; float64 intermediate math is soft-float on ESP32.

## Tests and evidence

- `go test -count=1 -json ./x/types/...` (2026-11-09, darwin/arm64): `x/types` 27 pass; `x/types/st` **build failed** (test file declares the amd64 asm symbol `convertInt16ToFloat32Assembly`, `convert_bench_test.go:583`); `x/types/internal` no tests.
- InfraNotes ran `x/types/st` with `GOARCH=amd64` under Rosetta: 18/18 pass (the asm placeholder is benchmarked, not value-checked). Probes: `~/src/tmp/easyrobot/tensor/InfraNotes/{conv,asmprobe}`.
- Cross-build (lead, compile only): linux/riscv64, linux/arm, linux/386 build `x/types/...`; js/wasm fails.

## API notes (ease of use, composability)

- Good: one place for saturating casts and strided copies; generic forms are zero-allocation and easy to call from kernels.
- Weak: `Convert` vs `Cast` naming is unclear (`Convert` is normalised audio-style scaling, `Cast` is a numeric cast); silent no-ops for unsupported type pairs hide bugs (eager `Copy` between unsupported dtypes does nothing); no uint8 although image data and int8 quantisation need it; no rounding mode parameter.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Replace GOARCH-list int-size files with `strconv.IntSize`/`math.MaxInt` logic — x/types/st/convert_i32.go:1, x/types/internal/constraints_i32.go:1 #p1 #ER-026

## Inferred objectives (proposals)

- **Proposal:** split into (a) typed, generic, allocation-free casts for kernels and (b) one dynamic dispatcher used only at API edges, returning errors; this removes most type switches from TinyGo builds.
- **Proposal:** add a `quant` sub-package (affine int8/uint8 quantise/dequantise with scale/zero-point, per-tensor and per-channel) as the base for W9 int8 kernels.
