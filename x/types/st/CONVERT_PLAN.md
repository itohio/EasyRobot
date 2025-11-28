# Convert Implementation Plan

## Overview
Create a new `Convert` function in `x/types/st/convert.go` that converts samples between different numeric types with proper scaling and dithering. This is different from `Cast` (which clamps) and `Copy` (which copies same types).

## Primary Use Case
Convert audio samples between formats:
- INT8/INT16/INT32 ↔ float32 or complex64(real component)
- Scaling from one type's range to another (e.g., float32 [-1,1] → int16 [-32768,32767])
- Dithering for downconversions to preserve audio quality

## Issues Identified in cmd/audio

### Repeated Code Patterns
1. **Scaling and clamping logic repeated** in multiple files:
   - `common.go:474-505` - `samplesToTensor()` converts float32 to INT16/INT32/UINT8
   - `gen_sine.go:115-154` - Similar conversion patterns
   - `gen_chirp.go:100-139` - Similar conversion patterns
   - `gen_noise.go:108-147` - Similar conversion patterns
   - `gen_dsss.go:446-528` - Similar conversion patterns
   - `gen_tc.go:159-190` - Similar conversion patterns
   - `record.go:429-504` - `writeTensorToEncoder()` converts from various types

2. **Magic numbers repeated**:
   - `32767.0` for int16 scaling
   - `2147483647.0` for int32 scaling
   - `127.5` for uint8 scaling
   - Manual clamping checks repeated everywhere

3. **No dithering**: All conversions use simple truncation/clamping, which can introduce quantization noise

## Design Requirements

### API Design
```go
// Convert converts src to dst with proper scaling between type ranges.
// For downconversions, applies dithering to reduce quantization artifacts.
// For same-type conversions, uses copy (fast path).
func Convert[T, U Number](dst []T, src []U, n int)

// ConvertAny converts src to dst with proper scaling (any types).
// For downconversions, applies dithering to reduce quantization artifacts.
// For same-type conversions, uses copy (fast path).
// Returns the destination slice.
func ConvertAny(dstData, srcData any) any

// ConvertStrided handles strided/non-contiguous arrays
func ConvertStrided[T, U Number](dst []T, src []U, shape []int, stridesDst, stridesSrc []int)

// ConvertStridedAny handles strided/non-contiguous arrays with any types
func ConvertStridedAny(dstData, srcData any, shape []int, stridesDst, stridesSrc []int) any
```

### Type Ranges (Audio Sample Ranges)
- **int8**: [-128, 127] → normalized to [-1.0, 1.0] for float32
- **int16**: [-32768, 32767] → normalized to [-1.0, 1.0] for float32
- **int32**: [-2147483648, 2147483647] → normalized to [-1.0, 1.0] for float32
- **int**: depends on architecture
- **float32**: [-1.0, 1.0] (normalized audio range)
- **complex64**: [-1.0, 1.0] (normalized real audio range)
- **uint8**: [0, 255] → normalized to [-1.0, 1.0] for float32 (offset by 128)

### Scaling Logic
1. **Same type**: Use `copy()` - fastest path
2. **Up-conversion** (e.g., int8 → int16): Scale from source range to destination range
3. **Down-conversion** (e.g., int16 → int8): Scale with dithering

### Dithering Strategy
- **Triangular PDF (TPDF) dithering** for downconversions
- Only apply dithering when reducing bit depth (downconversion)
- Use efficient PRNG (Xorshift or similar) for dither noise
- Dither range: ±1 LSB of destination type

### Performance Requirements
1. **Type dispatch**: Separate functions for each type combination (like cast.go)
2. **No type casts in hot path loops**
3. **BCE optimizations**: Use `for i := range n`, reslice to exact size
4. **Stack arrays**: Use stack-allocated arrays for small scratch space
5. **Loop unrolling**: For hot paths, unroll inner loops by 4-8
6. **Same-type fast path**: Direct copy, no conversion overhead
7. **single conversion function per type pair**: each conversion implemented separately in a highly optimized way

## Implementation Structure

### File Organization
```
x/types/st/convert.go         - Main dispatch and public API
x/types/st/convert_ops.go     - unexported optimized type-pair converters
x/types/st/convert_i32.go     - for architectures where int is int32
x/types/st/convert_i64.go     - for architectures where int is int64
x/types/st/convert_dither.go  - Dithering utilities
x/types/st/convert_test.go    - Unit tests
x/types/st/convert_bench_test.go - Benchmarks
```

### Type Dispatch Pattern (similar to cast.go)
```go
func Convert[T, U Number](dst []T, src []U, n int) {
    // Fast path: same type
    if sameType(dst, src) {
        copy(dst[:n], src[:n])
        return
    }
    
    // Dispatch to specialized functions
    switch d := any(dst).(type) {
    case []int8:
        convertToInt8(d, src, n)
    case []int16:
        convertToInt16(d, src, n)
    case []int32:
        convertToInt32(d, src, n)
    case []float32:
        convertToFloat32(d, src, n)
    // ... etc
    }
}
```

### Conversion Functions Needed

#### int8 Conversions
- `convertToInt8FromFloat32` - with dithering
- `convertToInt8FromInt16` - with dithering
- `convertToInt8FromInt32` - with dithering
- `convertFromInt8ToFloat32` - scaling only
- `convertFromInt8ToInt16` - scaling only
- `convertFromInt8ToInt32` - scaling only

#### int16 Conversions
- `convertToInt16FromFloat32` - with dithering (if needed)
- `convertToInt16FromInt32` - with dithering
- `convertFromInt16ToFloat32` - scaling only
- `convertFromInt16ToInt32` - scaling only
- `convertFromInt16ToInt8` - with dithering

#### int32 Conversions
- `convertToInt32FromFloat32` - scaling only (no dithering needed)
- `convertFromInt32ToFloat32` - scaling only
- `convertFromInt32ToInt16` - with dithering
- `convertFromInt32ToInt8` - with dithering

#### float32 Conversions
- `convertToFloat32FromInt8` - scaling only
- `convertToFloat32FromInt16` - scaling only
- `convertToFloat32FromInt32` - scaling only
- `convertFromFloat32ToInt8` - with dithering
- `convertFromFloat32ToInt16` - with dithering
- `convertFromFloat32ToInt32` - scaling only (no dithering needed)

## Implementation Steps

### Phase 1: Core Infrastructure
- [ ] Create convert.go with main dispatch function
- [ ] Implement type range constants and scaling factors
- [ ] Implement dithering utilities (TPDF dithering)
- [ ] Implement same-type fast path detection

### Phase 2: Type-Specific Conversions
- [ ] Implement int8 conversion functions (to/from all types)
- [ ] Implement int16 conversion functions (to/from all types)
- [ ] Implement int32 conversion functions (to/from all types)
- [ ] Implement float32 conversion functions (to/from all types)

### Phase 3: Optimization
- [ ] Apply BCE optimizations (for range n, reslicing)
- [ ] Remove type casts from hot paths
- [ ] Add loop unrolling for hot paths
- [ ] Use stack arrays for scratch space

### Phase 4: Strided Support
- [ ] Implement ConvertStrided function
- [ ] Reuse AdvanceOffsets pattern from cast.go
- [ ] Optimize strided paths

### Phase 5: Testing
- [ ] Unit tests for all type combinations
- [ ] Edge case tests (boundary values, empty slices, etc.)
- [ ] Dithering quality tests (verify noise shaping)
- [ ] Round-trip tests (convert A→B→A and verify)

### Phase 6: Benchmarks
- [ ] Benchmark same-type conversions (should match copy performance)
- [ ] Benchmark up-conversions
- [ ] Benchmark down-conversions with/without dithering
- [ ] Compare against existing cmd/audio inline code

### Phase 7: Refactoring
- [ ] Replace cmd/audio conversion code with Convert calls
- [ ] Remove repeated scaling/clamping logic
- [ ] Verify audio quality maintained or improved

## Actionable Task Backlog

### Phase 1: Preparation & Infrastructure

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P1.1 | Check existing convert* functions and rename to cast* | ⏳ Pending | `x/types/convert*.go` | All convert* functions renamed, no conflicts |
| P1.2 | Create convert.go with package declaration and imports | ✅ Complete | `x/types/st/convert.go` | File exists, imports Number from internal |
| P1.3 | Define type range constants (MaxInt8, MaxInt16, MaxInt32, etc.) | ✅ Complete | `x/types/st/convert_ops.go` | Constants defined for all audio types |
| P1.4 | Define scaling factor constants (ScaleInt8ToFloat32, etc.) | ✅ Complete | `x/types/st/convert_ops.go` | All scaling factors precomputed as constants |
| P1.5 | Create convert_dither.go with Xorshift64* PRNG | ✅ Complete | `x/types/st/convert_dither.go` | PRNG implemented, tested for uniform distribution |
| P1.6 | Implement TPDF dithering function (two uniform randoms, subtract) | ✅ Complete | `x/types/st/convert_dither.go` | Function returns triangular PDF noise in [-1, 1] range |
| P1.7 | Implement sameType() helper function | ✅ Complete | `x/types/st/convert.go` | Detects same type using type assertions, returns bool |
| P1.8 | Create main Convert() function with type dispatch skeleton | ✅ Complete | `x/types/st/convert.go` | Function signature matches API, dispatches to convertTo* functions |
| P1.9 | Implement ConvertAny() function (any types, returns dst) | ✅ Complete | `x/types/st/convert_any.go` | Similar to CastClampedAny, uses Convert internally |
| P1.10 | Implement ConvertStridedAny() function (any types, strided) | ✅ Complete | `x/types/st/convert_any.go` | Similar to CastClampedStridedAny, uses ConvertStrided internally |

### Phase 2: Core Conversion Functions - int8

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P2.1 | Implement convertToInt8() dispatcher function | ✅ Complete | `x/types/st/convert_ops.go` | Switches on source type, calls specific converters |
| P2.2 | Implement convertToInt8FromFloat32 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-1,1] to [-128,127], applies TPDF dither, uses for range n |
| P2.3 | Implement convertToInt8FromInt16 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-32768,32767] to [-128,127], applies dither, optimized loop |
| P2.4 | Implement convertToInt8FromInt32 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Scales int32 range to int8, applies dither, no type casts in loop |
| P2.5 | Implement convertToInt8FromComplex64 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Extracts real component, scales to [-128,127], applies dither |
| P2.6 | Implement convertFromInt8ToFloat32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-128,127] to [-1,1], no dithering, BCE optimized |
| P2.7 | Implement convertFromInt8ToInt16 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-128,127] to [-32768,32767], up-conversion |
| P2.8 | Implement convertFromInt8ToInt32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-128,127] to int32 range, up-conversion |
| P2.9 | Implement convertFromInt8ToComplex64 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-128,127] to [-1,1], creates complex64(real, 0) |

### Phase 3: Core Conversion Functions - int16

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P3.1 | Implement convertToInt16() dispatcher function | ✅ Complete | `x/types/st/convert_ops.go` | Switches on source type, calls specific converters |
| P3.2 | Implement convertToInt16FromFloat32 (no dithering needed) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-1,1] to [-32768,32767], clamps, optimized |
| P3.3 | Implement convertToInt16FromInt32 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Scales int32 to int16, applies dither, optimized loop |
| P3.4 | Implement convertToInt16FromComplex64 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Extracts real component, scales to [-32768,32767], applies dither |
| P3.5 | Implement convertFromInt16ToFloat32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-32768,32767] to [-1,1], no dithering |
| P3.6 | Implement convertFromInt16ToInt32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales int16 to int32, up-conversion |
| P3.7 | Implement convertFromInt16ToInt8 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt8FromInt16 (redundant wrapper removed) |
| P3.8 | Implement convertFromInt16ToComplex64 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-32768,32767] to [-1,1], creates complex64(real, 0) |

### Phase 4: Core Conversion Functions - int32

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P4.1 | Implement convertToInt32() dispatcher function | ✅ Complete | `x/types/st/convert_ops.go` | Switches on source type, calls specific converters |
| P4.2 | Implement convertToInt32FromFloat32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales [-1,1] to int32 range, clamps, no dithering |
| P4.3 | Implement convertToInt32FromComplex64 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Extracts real component, scales to int32 range, no dithering |
| P4.4 | Implement convertFromInt32ToFloat32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales int32 to [-1,1], no dithering |
| P4.5 | Implement convertFromInt32ToInt16 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt16FromInt32 (redundant wrapper removed) |
| P4.6 | Implement convertFromInt32ToInt8 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt8FromInt32 (redundant wrapper removed) |
| P4.7 | Implement convertFromInt32ToComplex64 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Scales int32 to [-1,1], creates complex64(real, 0) |

### Phase 5: Core Conversion Functions - float32

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P5.1 | Implement convertToFloat32() dispatcher function | ✅ Complete | `x/types/st/convert_ops.go` | Switches on source type, calls specific converters |
| P5.2 | Implement convertToFloat32FromInt8 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromInt8ToFloat32 |
| P5.3 | Implement convertToFloat32FromInt16 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromInt16ToFloat32 |
| P5.4 | Implement convertToFloat32FromInt32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromInt32ToFloat32 |
| P5.5 | Implement convertToFloat32FromComplex64 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromComplex64ToFloat32 |
| P5.6 | Implement convertFromFloat32ToInt8 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt8FromFloat32 (redundant wrapper removed) |
| P5.7 | Implement convertFromFloat32ToInt16 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt16FromFloat32 (redundant wrapper removed) |
| P5.8 | Implement convertFromFloat32ToInt32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt32FromFloat32 (redundant wrapper removed) |
| P5.9 | Implement convertFromFloat32ToComplex64 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Creates complex64(val, 0) from float32 |

### Phase 6: Architecture-Specific int Conversions

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P6.1 | Create convert_i32.go with build tag for 32-bit archs | ✅ Complete | `x/types/st/convert_i32.go` | File with build tag, handles int as int32 |
| P6.2 | Implement convertToInt() for 32-bit (int = int32) | ✅ Complete | `x/types/st/convert_i32.go` | Dispatcher for int conversions on 32-bit |
| P6.3 | Implement convertFromInt() for 32-bit (int = int32) | ✅ Complete | `x/types/st/convert_i32.go` | All int conversion paths for 32-bit |
| P6.4 | Create convert_i64.go with build tag for 64-bit archs | ✅ Complete | `x/types/st/convert_i64.go` | File with build tag, handles int as int64 |
| P6.5 | Implement convertToInt() for 64-bit (int = int64) | ✅ Complete | `x/types/st/convert_i64.go` | Dispatcher for int conversions on 64-bit |
| P6.6 | Implement convertFromInt() for 64-bit (int = int64) | ✅ Complete | `x/types/st/convert_i64.go` | All int conversion paths for 64-bit |

### Phase 7: Complex64 Support

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P7.1 | Implement convertToComplex64() dispatcher | ✅ Complete | `x/types/st/convert_ops.go` | Switches on source type, calls specific converters to complex64 |
| P7.2 | Implement convertFromComplex64() dispatcher | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromComplex64ToFloat32, integrated in dispatchers |
| P7.3 | Implement convertToComplex64FromInt8 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromInt8ToComplex64 |
| P7.4 | Implement convertToComplex64FromInt16 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromInt16ToComplex64 |
| P7.5 | Implement convertToComplex64FromInt32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromInt32ToComplex64 |
| P7.6 | Implement convertToComplex64FromFloat32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertFromFloat32ToComplex64 |
| P7.7 | Implement convertFromComplex64ToInt8 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt8FromComplex64 (redundant wrapper removed) |
| P7.8 | Implement convertFromComplex64ToInt16 with dithering | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt16FromComplex64 (redundant wrapper removed) |
| P7.9 | Implement convertFromComplex64ToInt32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses convertToInt32FromComplex64 (redundant wrapper removed) |
| P7.10 | Implement convertFromComplex64ToFloat32 (scaling only) | ✅ Complete | `x/types/st/convert_ops.go` | Extracts real() component, already in [-1,1] range |

### Phase 8: Optimization Pass

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P8.1 | Replace all `for i := 0; i < n; i++` with `for i := range n` | ✅ Complete | All convert files | All loops use range syntax for BCE (Go 1.22+) |
| P8.2 | Add BCE hints: reslice to exact size before loops | ✅ Complete | All convert files | `dst = dst[:n]`, `src = src[:n]` before loops (tells compiler exact bounds) |
| P8.3 | Remove all type casts from hot path loops | ✅ Complete | All convert files | Type assertions only in dispatchers, not loops |
| P8.4 | Use stack arrays for dither PRNG state | ✅ Complete | `x/types/st/convert_dither.go` | PRNG uses struct, no heap allocations |
| P8.5 | Use stack arrays for temporary indices/offsets (strided only) | ✅ Complete | `x/types/st/convert_ops.go` | Uses `var indicesStatic [MAX_DIMS]int` pattern for strided |
| P8.6 | Verify zero allocations in hot paths (go test -bench) | ⏳ Pending | All convert files | Benchmarks show 0 allocs/op for hot paths |
| P8.7 | Add bounds check elimination hints: `_ = dst[n-1]` before loops | ✅ Complete | All convert files | Reslicing used instead, which is more effective |

### Phase 9: Strided Support

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P9.1 | Implement ConvertStrided() public function | ✅ Complete | `x/types/st/convert.go` | Function signature matches API design |
| P9.2 | Implement strided dispatcher (similar to cast.go) | ✅ Complete | `x/types/st/convert_ops.go` | Uses AdvanceOffsets pattern from cast.go |
| P9.3 | Implement strided versions of all int8 conversions | ✅ Complete | `x/types/st/convert_ops.go` | convertStridedToInt8 implemented |
| P9.4 | Implement strided versions of all int16 conversions | ✅ Complete | `x/types/st/convert_ops.go` | convertStridedToInt16 implemented |
| P9.5 | Implement strided versions of all int32 conversions | ✅ Complete | `x/types/st/convert_ops.go` | convertStridedToInt32 implemented |
| P9.6 | Implement strided versions of all float32 conversions | ✅ Complete | `x/types/st/convert_ops.go` | convertStridedToFloat32 implemented |
| P9.7 | Implement strided versions of all complex64 conversions | ✅ Complete | `x/types/st/convert_ops.go` | convertStridedToComplex64 implemented |

### Phase 10: Unit Tests

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P10.1 | Create convert_test.go with test structure | ✅ Complete | `x/types/st/convert_test.go` | File exists, uses standard testing package |
| P10.2 | Test same-type fast path (int8→int8, int16→int16, etc.) | ✅ Complete | `x/types/st/convert_test.go` | Verifies copy() is used, matches input exactly |
| P10.3 | Test float32 → int16 conversion (scaling + dithering) | ✅ Complete | `x/types/st/convert_test.go` | Verifies [-1,1] maps to [-32768,32767], dither applied |
| P10.4 | Test int16 → float32 conversion (scaling only) | ✅ Complete | `x/types/st/convert_test.go` | Verifies [-32768,32767] maps to [-1,1] |
| P10.5 | Test float32 → int8 conversion (scaling + dithering) | ✅ Complete | `x/types/st/convert_test.go` | Verifies scaling and dithering for downconversion |
| P10.6 | Test int8 → float32 conversion (scaling only) | ✅ Complete | `x/types/st/convert_test.go` | Verifies up-conversion scaling |
| P10.7 | Test int32 → int16 conversion (scaling + dithering) | ✅ Complete | `x/types/st/convert_test.go` | Verifies downconversion with dithering |
| P10.8 | Test int16 → int32 conversion (scaling only) | ✅ Complete | `x/types/st/convert_test.go` | Verifies up-conversion |
| P10.9 | Test edge cases: empty slices, boundary values | ✅ Complete | `x/types/st/convert_test.go` | Tests -1.0, 1.0, -128, 127, -32768, 32767, etc. |
| P10.10 | Test dithering quality (verify TPDF distribution) | ✅ Complete | `x/types/st/convert_test.go` | Statistical test confirms triangular distribution |
| P10.11 | Test round-trip conversions (A→B→A) | ✅ Complete | `x/types/st/convert_test.go` | float32→int16→float32 preserves signal within tolerance |
| P10.12 | Test strided conversions | ✅ Complete | `x/types/st/convert_test.go` | All strided paths tested with various shapes/strides |
| P10.13 | Test complex64 ↔ float32 conversions | ✅ Complete | `x/types/st/convert_test.go` | complex64 ↔ float32 conversions work correctly |
| P10.14 | Test complex64 ↔ int8 conversions (with dithering) | ✅ Complete | `x/types/st/convert_test.go` | complex64 ↔ int8 with proper scaling and dithering |
| P10.15 | Test complex64 ↔ int16 conversions (with dithering) | ✅ Complete | `x/types/st/convert_test.go` | complex64 ↔ int16 with proper scaling and dithering |
| P10.16 | Test complex64 ↔ int32 conversions (scaling only) | ✅ Complete | `x/types/st/convert_test.go` | complex64 ↔ int32 with proper scaling, no dithering |

### Phase 11: Benchmarks

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P11.1 | Create convert_bench_test.go | ✅ Complete | `x/types/st/convert_bench_test.go` | File exists with benchmark structure |
| P11.2 | Benchmark same-type conversions (int8→int8, etc.) | ✅ Complete | `x/types/st/convert_bench_test.go` | Benchmarks created, includes copy() baseline for comparison |
| P11.3 | Benchmark float32 → int16 (with dithering) | ✅ Complete | `x/types/st/convert_bench_test.go` | Benchmark created, includes inline baseline comparison |
| P11.4 | Benchmark int16 → float32 (scaling only) | ✅ Complete | `x/types/st/convert_bench_test.go` | Benchmark created, includes inline baseline comparison |
| P11.5 | Benchmark float32 → int8 (with dithering) | ✅ Complete | `x/types/st/convert_bench_test.go` | Benchmark created, includes inline baseline comparison |
| P11.6 | Benchmark strided conversions | ✅ Complete | `x/types/st/convert_bench_test.go` | Benchmarks for contiguous and non-contiguous strides |
| P11.7 | Benchmark all type combinations | ✅ Complete | `x/types/st/convert_bench_test.go` | Complete coverage: int8, int16, int32, float32, complex64 conversions |

### Phase 12: Refactoring cmd/audio

| ID | Task | Status | File | Acceptance Criteria |
|----|------|--------|------|-------------------|
| P12.1 | Replace samplesToTensor() conversion in common.go | ⏳ Pending | `cmd/audio/common.go` | Uses st.Convert() instead of inline code |
| P12.2 | Replace conversions in gen_sine.go | ⏳ Pending | `cmd/audio/gen_sine.go` | Uses st.Convert() for all type conversions |
| P12.3 | Replace conversions in gen_chirp.go | ⏳ Pending | `cmd/audio/gen_chirp.go` | Uses st.Convert() for all type conversions |
| P12.4 | Replace conversions in gen_noise.go | ⏳ Pending | `cmd/audio/gen_noise.go` | Uses st.Convert() for all type conversions |
| P12.5 | Replace conversions in gen_dsss.go | ⏳ Pending | `cmd/audio/gen_dsss.go` | Uses st.Convert() for all type conversions |
| P12.6 | Replace conversions in gen_tc.go | ⏳ Pending | `cmd/audio/gen_tc.go` | Uses st.Convert() for all type conversions |
| P12.7 | Replace writeTensorToEncoder() conversions in record.go | ⏳ Pending | `cmd/audio/record.go` | Uses st.Convert() for tensor→int conversions |
| P12.8 | Remove all magic numbers (32767.0, 2147483647.0, 127.5) | ⏳ Pending | All cmd/audio files | All magic numbers removed, using Convert() |
| P12.9 | Verify audio quality maintained (manual testing) | ⏳ Pending | N/A | Audio output quality same or better with dithering |
| P12.10 | Run full test suite for cmd/audio | ⏳ Pending | `cmd/audio/` | All tests pass after refactoring |

### Status Legend
- ⏳ Pending - Not started
- 🔄 In Progress - Currently working on
- ✅ Complete - Finished and verified
- ❌ Blocked - Blocked by dependency or issue

## Notes

**IMPORTANT** There already exist a set of convert* functions -- these are not for samples conversion. they must be renamed to cast instead.

### Dithering Implementation Details
- Use Xorshift64* PRNG for dither noise (fast, good quality)
- TPDF: Generate two uniform random numbers, subtract to get triangular distribution
- Dither range: ±1 LSB of destination type
- Only apply for downconversions (reducing bit depth)

### Scaling Formulas

#### Integer to Float32 (Up-conversion, scaling only)
- **int8 → float32**: `val / 127.0` (normalize [-128, 127] to [-1.0, 1.0])
- **int16 → float32**: `val / 32767.0` (normalize [-32768, 32767] to [-1.0, 1.0])
- **int32 → float32**: `val / 2147483647.0` (normalize int32 range to [-1.0, 1.0])

#### Float32 to Integer (Down-conversion, with dithering)
- **float32 → int8**: `val * 127.0 + dither` (scale [-1.0, 1.0] to [-128, 127], clamp, apply TPDF dither)
- **float32 → int16**: `val * 32767.0 + dither` (scale [-1.0, 1.0] to [-32768, 32767], clamp, apply TPDF dither)
- **float32 → int32**: `val * 2147483647.0` (scale [-1.0, 1.0] to int32 range, clamp, no dithering needed)

#### Integer to Integer (Up/Down conversion)
- **int8 → int16**: `val * 256` (scale [-128, 127] to [-32768, 32767])
- **int16 → int8**: `val / 256 + dither` (scale [-32768, 32767] to [-128, 127], apply dither)
- **int8 → int32**: `val * 16777216` (scale [-128, 127] to int32 range)
- **int32 → int8**: `val / 16777216 + dither` (scale int32 to [-128, 127], apply dither)
- **int16 → int32**: `val * 65536` (scale [-32768, 32767] to int32 range)
- **int32 → int16**: `val / 65536 + dither` (scale int32 to [-32768, 32767], apply dither)

#### Complex64 Conversions (uses real component)
- **complex64 → float32**: `real(val)` (extract real component, already in [-1.0, 1.0])
- **float32 → complex64**: `complex(val, 0)` (create complex64 with real component, imag=0)
- **complex64 → int8**: `real(val) * 127.0 + dither` (extract real, scale, apply dither)
- **int8 → complex64**: `complex(val / 127.0, 0)` (scale to [-1,1], create complex64)
- **complex64 → int16**: `real(val) * 32767.0 + dither` (extract real, scale, apply dither)
- **int16 → complex64**: `complex(val / 32767.0, 0)` (scale to [-1,1], create complex64)
- **complex64 → int32**: `real(val) * 2147483647.0` (extract real, scale, no dithering)
- **int32 → complex64**: `complex(val / 2147483647.0, 0)` (scale to [-1,1], create complex64)

#### Uint8 Conversions (offset by 128)
- **float32 → uint8**: `(val + 1.0) * 127.5` (map [-1,1] to [0,255], clamp)
- **uint8 → float32**: `(val / 127.5) - 1.0` (map [0,255] to [-1,1])

### Performance Targets
- Same-type: Should match `copy()` performance (within 5%)
- Up-conversion: < 2x overhead vs direct cast
- Down-conversion with dithering: < 3x overhead vs direct cast
- Strided: < 2x overhead vs contiguous

## Loop Optimization Techniques (from BCE_ANALYSIS.md)

### Critical Optimization Patterns

**Note:** All slices are already flat (1D). No 2D operations or flattening needed. Only strided versions may use shape information for multi-dimensional access patterns.

#### 1. **Reslice for BCE (Best for Contiguous Operations)**
**Performance Gain: Eliminates bounds checks**

```go
// ✅ BEST: Reslice to exact size for contiguous flat slices
n := min(len(dst), len(src))
dst = dst[:n]  // BCE hint: compiler knows exact bounds
src = src[:n]  // BCE hint: compiler knows exact bounds

for i := range n {
    dst[i] = convert(src[i])  // No bounds checks!
}
```

**Why it works:**
- Reslicing tells compiler exact bounds (BCE)
- Eliminates bounds checks in loop
- Simple and effective for flat slices

**Apply to:** All contiguous conversion operations

#### 2. **Strided Operations with Shape Information (if needed)**
**Performance Gain: Better cache locality for multi-dimensional strided access**

```go
// ✅ GOOD: For strided operations that need shape info
// Use AdvanceOffsets pattern from cast.go for multi-dimensional strided access
var indicesStatic [MAX_DIMS]int
var offsetsStatic [2]int
indices := indicesStatic[:ndims]
offsets := offsetsStatic[:2]

for {
    dst[offsets[0]] = convert(src[offsets[1]])
    if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
        break
    }
}
```

**Why it works:**
- Handles multi-dimensional strided access efficiently
- Uses stack arrays for indices/offsets (no heap allocation)
- AdvanceOffsets pattern is proven in cast.go

**Apply to:** Strided conversion operations that need shape information

#### 3. **Use `for range n` Instead of `for i := 0; i < n; i++`**
**Performance Gain: 2-3% improvement consistently**

```go
// ✅ GOOD: for range n (Go 1.22+)
for i := range n {
    dst[i] = convert(src[i])
}

// ❌ BAD: Traditional loop
for i := 0; i < n; i++ {
    dst[i] = convert(src[i])
}
```

**Why it works:**
- Compiler optimizes `for range n` better
- Better BCE opportunities
- Slightly faster

**Apply to:** All loops in conversion functions

#### 4. **Reslicing for BCE Hints**
**Performance Gain: Eliminates bounds checks**

```go
// ✅ BEST: Reslice to exact size
n := min(len(dst), len(src))
dst = dst[:n]  // BCE hint: compiler knows exact bounds
src = src[:n]  // BCE hint: compiler knows exact bounds

for i := range n {
    dst[i] = convert(src[i])  // No bounds checks!
}

// ✅ GOOD: Access last element as hint (when reslicing not possible)
if n > 0 {
    _ = dst[n-1]  // BCE hint
    _ = src[n-1]  // BCE hint
}
```

**Why it works:**
- `dst = dst[:n]` tells compiler the exact length
- Eliminates bounds checks in loop
- More effective than `_ = dst[n-1]` hints

**Apply to:** All conversion functions before loops

#### 5. **Stack Arrays Instead of `make()` for Small Scratch Space**
**Performance Gain: Zero allocations, faster execution**

```go
// ✅ GOOD: Stack allocation for small arrays
const MAX_DIMS = 16

var indicesStatic [MAX_DIMS]int
indices := indicesStatic[:ndims]  // Stack allocation!

var offsetsStatic [2]int
offsets := offsetsStatic[:2]  // Stack allocation!

// ❌ BAD: Heap allocation
indices := make([]int, ndims)  // Heap allocation!
offsets := make([]int, 2)      // Heap allocation!
```

**Why it works:**
- Stack allocation is much faster than heap
- No GC pressure
- Better cache locality
- Zero allocations in benchmarks

**Apply to:** All temporary arrays (shape, strides, indices, offsets)

**When to use `make()`:**
- Only for large arrays (> 1KB) that would overflow stack
- Only when size is dynamically determined and too large for stack
- For arrays that need to escape function scope


#### 7. **Avoid Type Casts in Hot Path Loops**
**Performance Gain: Eliminates conversion overhead**

```go
// ✅ GOOD: Type assertion in dispatcher, not in loop
func convertToInt8FromFloat32(dst []int8, src []float32, n int) {
    dst = dst[:n]  // BCE hint
    src = src[:n]  // BCE hint
    for i := range n {
        dst[i] = int8(src[i] * 127.0 + dither)  // No type assertions!
    }
}

// ❌ BAD: Type assertion in loop
func convertToInt8(dst []int8, src interface{}, n int) {
    srcFloat := src.([]float32)  // Type assertion
    for i := 0; i < n; i++ {
        dst[i] = int8(srcFloat[i] * 127.0)  // Type assertion overhead
    }
}
```

**Why it works:**
- Type assertions have overhead
- Move type dispatch to dispatcher functions
- Hot path loops only do conversions, not type checks

**Apply to:** All conversion functions

### Complete Optimization Checklist for Convert Functions

#### For Contiguous Flat Operations (1D slices):
1. ✅ **Reslice** both slices (`dst = dst[:n]`, `src = src[:n]`)
2. ✅ **Use `for i := range n`** for loop
3. ✅ **No type casts** in loop (type dispatch in dispatcher)
4. ✅ **Use stack arrays** for temporary data (dither state, etc.)

#### For Strided Operations:
1. ✅ **Use AdvanceOffsets pattern** from cast.go for multi-dimensional strided access
2. ✅ **Use stack arrays** for indices/offsets (`var indicesStatic [MAX_DIMS]int`)
3. ✅ **Use `for range`** loops where applicable
4. ✅ **Shape information** only used if needed for strided access patterns

#### Memory Management:
1. ✅ **Never use `make([]int, n)`** for shape/stride/indices (use `[MAX_DIMS]int`)
2. ✅ **Use stack arrays** for small scratch space (< 1KB)
3. ✅ **Zero allocations** in hot paths (verify with `go test -bench -benchmem`)

### Performance Targets (Based on BCE_ANALYSIS.md)

**Element-wise Operations (1M elements, flat slices):**
- **Best Go (Reslice+Range)**: ~2.06ms (38% faster than baseline)
- **Baseline**: ~3.26ms (no optimizations)

**Strided Operations (1M elements):**
- **Good (Reslice+Range)**: ~3.00ms (10% faster than strided baseline)
- **Strided Baseline**: ~3.35ms (with strides, no optimizations)

**For Convert Functions:**
- **Same-type**: Should match `copy()` performance (within 5%)
- **Up-conversion**: < 2x overhead vs direct cast
- **Down-conversion with dithering**: < 3x overhead vs direct cast
- **Strided**: < 2x overhead vs contiguous

