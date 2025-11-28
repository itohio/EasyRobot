# Audio Conversion Code Analysis

## Issues Identified

### 1. Repeated Conversion Logic

#### Issue: Scaling and Clamping Code Duplicated Across Multiple Files

**Location 1: `common.go` - `samplesToTensor()` function (lines 456-512)**
```go
case types.INT16:
    data := t.Data().([]int16)
    for i, v := range samples {
        val := v * 32767.0
        if val > 32767.0 {
            val = 32767.0
        } else if val < -32768.0 {
            val = -32768.0
        }
        data[i] = int16(val)
    }
case types.INT32:
    data := t.Data().([]int32)
    for i, v := range samples {
        val := v * 2147483647.0
        if val > 2147483647.0 {
            val = 2147483647.0
        } else if val < -2147483648.0 {
            val = -2147483648.0
        }
        data[i] = int32(val)
    }
case types.UINT8:
    data := t.Data().([]uint8)
    for i, v := range samples {
        val := (v + 1.0) * 127.5
        if val > 255.0 {
            val = 255.0
        } else if val < 0.0 {
            val = 0.0
        }
        data[i] = uint8(val)
    }
```

**Location 2: `gen_sine.go` (lines 115-154)**
- Identical scaling logic for INT16, INT32, UINT8
- Same magic numbers: 32767.0, 2147483647.0, 127.5

**Location 3: `gen_chirp.go` (lines 100-139)**
- Identical scaling logic repeated

**Location 4: `gen_noise.go` (lines 108-147)**
- Identical scaling logic repeated

**Location 5: `gen_dsss.go` (lines 446-528)**
- Identical scaling logic repeated

**Location 6: `gen_tc.go` (lines 159-190)**
- Identical scaling logic repeated

**Location 7: `record.go` - `writeTensorToEncoder()` (lines 429-504)**
- Reverse conversion logic (from tensor types to int for WAV encoding)
- Similar clamping patterns

### 2. Magic Numbers Repeated

The following magic numbers appear in multiple files:
- `32767.0` - int16 maximum value for scaling
- `2147483647.0` - int32 maximum value for scaling  
- `127.5` - uint8 scaling factor
- `-32768.0` / `-2147483648.0` - minimum values for clamping

**Impact**: Hard to maintain, error-prone if values need to change

### 3. No Dithering for Downconversions

**Issue**: All conversions use simple truncation/clamping, which introduces quantization noise.

**Example from `common.go`**:
```go
val := v * 32767.0
if val > 32767.0 {
    val = 32767.0
} else if val < -32768.0 {
    val = -32768.0
}
data[i] = int16(val)  // Simple truncation, no dithering
```

**Impact**: 
- Quantization artifacts in audio
- Reduced audio quality, especially for quiet signals
- Professional audio software uses dithering for bit depth reduction

### 4. Inefficient Type Conversions

**Issue**: Type assertions and conversions in hot paths

**Example from `record.go`**:
```go
switch t.DataType() {
case tensortypes.INT16:
    data := t.Data().([]int16)  // Type assertion in hot path
    for frame := 0; frame < frames; frame++ {
        for ch := 0; ch < channels; ch++ {
            intBuffer.Data[ch] = int(data[frame*channels+ch])  // Type conversion in loop
        }
    }
}
```

**Impact**: Performance overhead, especially for large audio buffers

### 5. Missing BCE Optimizations

**Issue**: Loops don't use modern Go optimizations

**Current pattern**:
```go
for i := 0; i < n; i++ {  // Old style
    // conversion logic
}
```

**Should be**:
```go
for i := range n {  // Better BCE, Go 1.22+
    // conversion logic
}
```

### 6. No Reuse of Existing Type Conversion Infrastructure

**Issue**: `x/types/st/cast.go` already has optimized type conversion infrastructure, but cmd/audio doesn't use it.

**Impact**: 
- Duplicated effort
- Inconsistent conversion behavior
- Missing optimizations from cast.go

## Recommended Solution

### Create `x/types/st/convert.go`

A new conversion system that:
1. **Scales between type ranges** (not just clamps like Cast)
2. **Uses dithering for downconversions** (audio quality)
3. **Reuses copy for same-type** (performance)
4. **Follows cast.go patterns** (type dispatch, optimization)
5. **Eliminates repeated code** in cmd/audio

### Benefits

1. **DRY**: Single source of truth for conversions
2. **Performance**: Optimized loops, BCE, no type casts in hot path
3. **Quality**: Dithering improves audio quality
4. **Maintainability**: Changes in one place
5. **Consistency**: Same conversion logic everywhere

## Refactoring Plan

### Files to Update

1. **`common.go`** - Replace `samplesToTensor()` conversion logic
2. **`gen_sine.go`** - Replace inline conversions
3. **`gen_chirp.go`** - Replace inline conversions
4. **`gen_noise.go`** - Replace inline conversions
5. **`gen_dsss.go`** - Replace inline conversions
6. **`gen_tc.go`** - Replace inline conversions
7. **`record.go`** - Replace `writeTensorToEncoder()` conversion logic

### Example Refactoring

**Before** (common.go):
```go
case types.INT16:
    data := t.Data().([]int16)
    for i, v := range samples {
        val := v * 32767.0
        if val > 32767.0 {
            val = 32767.0
        } else if val < -32768.0 {
            val = -32768.0
        }
        data[i] = int16(val)
    }
```

**After**:
```go
case types.INT16:
    data := t.Data().([]int16)
    st.Convert(data, samples, len(samples))
```

Much cleaner, and includes dithering for better quality!

## Summary

- **6 files** with repeated conversion code
- **~200 lines** of duplicated logic
- **No dithering** for audio quality
- **Inefficient** type conversions
- **Missing optimizations** (BCE, modern Go patterns)

**Solution**: Create `x/types/st/convert.go` with optimized, dithered conversions and refactor cmd/audio to use it.

