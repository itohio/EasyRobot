package st

import (
	. "github.com/itohio/EasyRobot/x/math/shape"
	. "github.com/itohio/EasyRobot/x/types/internal"

	"math"
)

// Type range constants for audio samples
const (
	// Scaling factors for conversion
	scaleInt8ToFloat32   = 1.0 / 127.0   // [-128, 127] -> [-1.0, 1.0]
	scaleInt16ToFloat32  = 1.0 / 32767.0 // [-32768, 32767] -> [-1.0, 1.0]
	scaleInt32ToFloat32  = 1.0 / 2147483647.0
	scaleFloat32ToInt8   = 127.0
	scaleFloat32ToInt16  = 32767.0
	scaleFloat32ToInt32  = 2147483647.0
)

// Function naming convention:
// - convertToXFromY: Converts TO destination type X FROM source type Y
//   Used in convertToX dispatcher for down-conversions and float32/complex64 sources
// - convertFromYToX: Converts FROM source type Y TO destination type X
//   Used in convertToX dispatcher for up-conversions (e.g., int8->int16, int16->int32)
// All functions are directly called by dispatchers - no redundant wrappers.

// convertToInt8 dispatches to specific int8 conversion functions
func convertToInt8[U Number](dst []int8, src []U, n int) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]

	switch s := any(src).(type) {
	case []int8:
		copy(dst, s)
	case []int16:
		convertToInt8FromInt16(dst, s)
	case []int32:
		convertToInt8FromInt32(dst, s)
	case []int:
		convertToInt8FromInt(dst, s)
	case []float32:
		convertToInt8FromFloat32(dst, s)
	case []complex64:
		convertToInt8FromComplex64(dst, s)
	}
}

// convertToInt8FromFloat32 converts float32 [-1,1] to int8 [-128,127] with dithering
func convertToInt8FromFloat32(dst []int8, src []float32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	// Initialize dither PRNG with seed based on first sample
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(math.Float32bits(src[0]))
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := src[i]
		// Scale [-1,1] to [-128,127] range
		scaled := val * scaleFloat32ToInt8
		// Apply TPDF dither (±1 LSB)
		dither := tpdfDither(rng) // [-1, 1]
		result := scaled + dither
		
		// Clamp to int8 range
		if result > 127.0 {
			dst[i] = 127
		} else if result < -128.0 {
			dst[i] = -128
		} else {
			dst[i] = int8(result)
		}
	}
}

// convertToInt8FromInt16 converts int16 to int8 with dithering
func convertToInt8FromInt16(dst []int8, src []int16) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	// Initialize dither PRNG
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(src[0]) + 0x10000
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := float32(src[i])
		// Scale from int16 range to int8 range
		scaled := val / 256.0 // Divide by 256 to scale down
		// Apply TPDF dither
		dither := tpdfDither(rng)
		result := scaled + dither
		
		// Clamp to int8 range
		if result > 127.0 {
			dst[i] = 127
		} else if result < -128.0 {
			dst[i] = -128
		} else {
			dst[i] = int8(result)
		}
	}
}

// convertToInt8FromInt32 converts int32 to int8 with dithering
func convertToInt8FromInt32(dst []int8, src []int32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	// Initialize dither PRNG
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(src[0]) + 0x100000000
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := float32(src[i])
		// Scale from int32 range to int8 range
		scaled := val / 16777216.0 // Divide by 2^24
		// Apply TPDF dither
		dither := tpdfDither(rng)
		result := scaled + dither
		
		// Clamp to int8 range
		if result > 127.0 {
			dst[i] = 127
		} else if result < -128.0 {
			dst[i] = -128
		} else {
			dst[i] = int8(result)
		}
	}
}

// convertToInt8FromInt is architecture-specific, see convert_i32.go and convert_i64.go

// convertToInt8FromComplex64 converts complex64 to int8 with dithering
func convertToInt8FromComplex64(dst []int8, src []complex64) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	// Initialize dither PRNG
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(math.Float32bits(real(src[0])))
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := real(src[i]) // Extract real component
		// Scale [-1,1] to [-128,127] range
		scaled := val * scaleFloat32ToInt8
		// Apply TPDF dither
		dither := tpdfDither(rng)
		result := scaled + dither
		
		// Clamp to int8 range
		if result > 127.0 {
			dst[i] = 127
		} else if result < -128.0 {
			dst[i] = -128
		} else {
			dst[i] = int8(result)
		}
	}
}

// convertFromInt8ToFloat32 converts int8 [-128,127] to float32 [-1,1] (scaling only)
func convertFromInt8ToFloat32(dst []float32, src []int8) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = float32(src[i]) * scaleInt8ToFloat32
	}
}

// convertFromInt8ToInt16 converts int8 to int16 (scaling only, up-conversion)
func convertFromInt8ToInt16(dst []int16, src []int8) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int16(src[i]) * 256 // Scale up by 256
	}
}

// convertFromInt8ToInt32 converts int8 to int32 (scaling only, up-conversion)
func convertFromInt8ToInt32(dst []int32, src []int8) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int32(src[i]) * 16777216 // Scale up by 2^24
	}
}

// convertFromInt8ToComplex64 converts int8 to complex64 (scaling only)
func convertFromInt8ToComplex64(dst []complex64, src []int8) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := float32(src[i]) * scaleInt8ToFloat32
		dst[i] = complex(val, 0)
	}
}

// convertToInt16 dispatches to specific int16 conversion functions
func convertToInt16[U Number](dst []int16, src []U, n int) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]

	switch s := any(src).(type) {
	case []int16:
		copy(dst, s)
	case []int8:
		convertFromInt8ToInt16(dst, s)
	case []int32:
		convertToInt16FromInt32(dst, s)
	case []int:
		convertToInt16FromInt(dst, s)
	case []float32:
		convertToInt16FromFloat32(dst, s)
	case []complex64:
		convertToInt16FromComplex64(dst, s)
	}
}

// convertToInt16FromFloat32 converts float32 [-1,1] to int16 [-32768,32767] (no dithering needed)
func convertToInt16FromFloat32(dst []int16, src []float32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := src[i] * scaleFloat32ToInt16
		if val > 32767.0 {
			dst[i] = 32767
		} else if val < -32768.0 {
			dst[i] = -32768
		} else {
			dst[i] = int16(val)
		}
	}
}

// convertToInt16FromInt32 converts int32 to int16 with dithering
func convertToInt16FromInt32(dst []int16, src []int32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	// Initialize dither PRNG
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(src[0]) + 0x300000000
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := float32(src[i])
		// Scale from int32 range to int16 range
		scaled := val / 65536.0 // Divide by 2^16
		// Apply TPDF dither
		dither := tpdfDither(rng)
		result := scaled + dither
		
		// Clamp to int16 range
		if result > 32767.0 {
			dst[i] = 32767
		} else if result < -32768.0 {
			dst[i] = -32768
		} else {
			dst[i] = int16(result)
		}
	}
}

// convertToInt16FromInt is architecture-specific, see convert_i32.go and convert_i64.go

// convertToInt16FromComplex64 converts complex64 to int16 with dithering
func convertToInt16FromComplex64(dst []int16, src []complex64) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	// Initialize dither PRNG
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(math.Float32bits(real(src[0]))) + 0x500000000
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := real(src[i]) // Extract real component
		// Scale [-1,1] to [-32768,32767] range
		scaled := val * scaleFloat32ToInt16
		// Apply TPDF dither
		dither := tpdfDither(rng)
		result := scaled + dither
		
		// Clamp to int16 range
		if result > 32767.0 {
			dst[i] = 32767
		} else if result < -32768.0 {
			dst[i] = -32768
		} else {
			dst[i] = int16(result)
		}
	}
}

// convertFromInt16ToFloat32 converts int16 [-32768,32767] to float32 [-1,1] (scaling only)
func convertFromInt16ToFloat32(dst []float32, src []int16) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = float32(src[i]) * scaleInt16ToFloat32
	}
}

// convertFromInt16ToInt32 converts int16 to int32 (scaling only, up-conversion)
func convertFromInt16ToInt32(dst []int32, src []int16) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int32(src[i]) * 65536 // Scale up by 2^16
	}
}


// convertFromInt16ToComplex64 converts int16 to complex64 (scaling only)
func convertFromInt16ToComplex64(dst []complex64, src []int16) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := float32(src[i]) * scaleInt16ToFloat32
		dst[i] = complex(val, 0)
	}
}

// convertToInt32 dispatches to specific int32 conversion functions
func convertToInt32[U Number](dst []int32, src []U, n int) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]

	switch s := any(src).(type) {
	case []int32:
		copy(dst, s)
	case []int8:
		convertFromInt8ToInt32(dst, s)
	case []int16:
		convertFromInt16ToInt32(dst, s)
	case []int:
		convertToInt32FromInt(dst, s)
	case []float32:
		convertToInt32FromFloat32(dst, s)
	case []complex64:
		convertToInt32FromComplex64(dst, s)
	}
}

// convertToInt32FromFloat32 converts float32 [-1,1] to int32 (scaling only, no dithering needed)
func convertToInt32FromFloat32(dst []int32, src []float32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := src[i] * scaleFloat32ToInt32
		if val > float32(math.MaxInt32) {
			dst[i] = math.MaxInt32
		} else if val < float32(math.MinInt32) {
			dst[i] = math.MinInt32
		} else {
			dst[i] = int32(val)
		}
	}
}

// convertToInt32FromInt is architecture-specific, see convert_i32.go and convert_i64.go

// convertToInt32FromComplex64 converts complex64 to int32 (scaling only)
func convertToInt32FromComplex64(dst []int32, src []complex64) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := real(src[i]) // Extract real component
		scaled := val * scaleFloat32ToInt32
		if scaled > float32(math.MaxInt32) {
			dst[i] = math.MaxInt32
		} else if scaled < float32(math.MinInt32) {
			dst[i] = math.MinInt32
		} else {
			dst[i] = int32(scaled)
		}
	}
}

// convertFromInt32ToFloat32 converts int32 to float32 [-1,1] (scaling only)
func convertFromInt32ToFloat32(dst []float32, src []int32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = float32(src[i]) * scaleInt32ToFloat32
	}
}


// convertFromInt32ToComplex64 converts int32 to complex64 (scaling only)
func convertFromInt32ToComplex64(dst []complex64, src []int32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := float32(src[i]) * scaleInt32ToFloat32
		dst[i] = complex(val, 0)
	}
}

// convertToFloat32 dispatches to specific float32 conversion functions
func convertToFloat32[U Number](dst []float32, src []U, n int) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]

	switch s := any(src).(type) {
	case []float32:
		copy(dst, s)
	case []int8:
		convertFromInt8ToFloat32(dst, s)
	case []int16:
		convertFromInt16ToFloat32(dst, s)
	case []int32:
		convertFromInt32ToFloat32(dst, s)
	case []int:
		convertFromIntToFloat32(dst, s)
	case []complex64:
		convertFromComplex64ToFloat32(dst, s)
	}
}

// convertFromIntToFloat32 is architecture-specific, see convert_i32.go and convert_i64.go

// convertFromComplex64ToFloat32 converts complex64 to float32 (extract real component)
func convertFromComplex64ToFloat32(dst []float32, src []complex64) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = real(src[i]) // Extract real component, already in [-1,1] range
	}
}


// convertFromFloat32ToComplex64 converts float32 to complex64
func convertFromFloat32ToComplex64(dst []complex64, src []float32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = complex(src[i], 0)
	}
}

// convertToComplex64 dispatches to specific complex64 conversion functions
func convertToComplex64[U Number](dst []complex64, src []U, n int) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]

	switch s := any(src).(type) {
	case []complex64:
		copy(dst, s)
	case []int8:
		convertFromInt8ToComplex64(dst, s)
	case []int16:
		convertFromInt16ToComplex64(dst, s)
	case []int32:
		convertFromInt32ToComplex64(dst, s)
	case []int:
		convertFromIntToComplex64(dst, s)
	case []float32:
		convertFromFloat32ToComplex64(dst, s)
	}
}

// convertFromIntToComplex64 is architecture-specific, see convert_i32.go and convert_i64.go


// convertStrided dispatches to strided conversion functions
func convertStrided[T, U Number](dst []T, src []U, shape []int, srcStrides, dstStrides []int) {
	// Use AdvanceOffsets pattern from cast.go
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:ndims]
	offsets := offsetsStatic[:2]

	// Dispatch based on destination type
	switch d := any(dst).(type) {
	case []int8:
		convertStridedToInt8(d, src, shape, indices, offsets, srcStrides, dstStrides)
	case []int16:
		convertStridedToInt16(d, src, shape, indices, offsets, srcStrides, dstStrides)
	case []int32:
		convertStridedToInt32(d, src, shape, indices, offsets, srcStrides, dstStrides)
	case []float32:
		convertStridedToFloat32(d, src, shape, indices, offsets, srcStrides, dstStrides)
	case []complex64:
		convertStridedToComplex64(d, src, shape, indices, offsets, srcStrides, dstStrides)
	}
}

// convertStridedToInt8 handles strided conversion to int8
func convertStridedToInt8[U Number](dst []int8, src []U, shape []int, indices, offsets []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []int8:
		for {
			dst[offsets[0]] = s[offsets[1]]
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	case []float32:
		rng := newXorshift64Star(0x123456789ABCDEF0)
		for {
			val := s[offsets[1]] * scaleFloat32ToInt8
			dither := tpdfDither(rng)
			result := val + dither
			if result > 127.0 {
				dst[offsets[0]] = 127
			} else if result < -128.0 {
				dst[offsets[0]] = -128
			} else {
				dst[offsets[0]] = int8(result)
			}
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	// Add other source types as needed
	}
}

// convertStridedToInt16 handles strided conversion to int16
func convertStridedToInt16[U Number](dst []int16, src []U, shape []int, indices, offsets []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []int16:
		for {
			dst[offsets[0]] = s[offsets[1]]
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	case []float32:
		for {
			val := s[offsets[1]] * scaleFloat32ToInt16
			if val > 32767.0 {
				dst[offsets[0]] = 32767
			} else if val < -32768.0 {
				dst[offsets[0]] = -32768
			} else {
				dst[offsets[0]] = int16(val)
			}
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	// Add other source types as needed
	}
}

// convertStridedToInt32 handles strided conversion to int32
func convertStridedToInt32[U Number](dst []int32, src []U, shape []int, indices, offsets []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []int32:
		for {
			dst[offsets[0]] = s[offsets[1]]
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	case []float32:
		for {
			val := s[offsets[1]] * scaleFloat32ToInt32
			if val > float32(math.MaxInt32) {
				dst[offsets[0]] = math.MaxInt32
			} else if val < float32(math.MinInt32) {
				dst[offsets[0]] = math.MinInt32
			} else {
				dst[offsets[0]] = int32(val)
			}
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	// Add other source types as needed
	}
}

// convertStridedToFloat32 handles strided conversion to float32
func convertStridedToFloat32[U Number](dst []float32, src []U, shape []int, indices, offsets []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []float32:
		for {
			dst[offsets[0]] = s[offsets[1]]
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	case []int16:
		for {
			dst[offsets[0]] = float32(s[offsets[1]]) * scaleInt16ToFloat32
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	// Add other source types as needed
	}
}

// convertStridedToComplex64 handles strided conversion to complex64
func convertStridedToComplex64[U Number](dst []complex64, src []U, shape []int, indices, offsets []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []complex64:
		for {
			dst[offsets[0]] = s[offsets[1]]
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	case []float32:
		for {
			dst[offsets[0]] = complex(s[offsets[1]], 0)
			if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
				break
			}
		}
	// Add other source types as needed
	}
}

