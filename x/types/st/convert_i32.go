//go:build 386 || arm || mips || mipsle

package st

import (
	. "github.com/itohio/EasyRobot/x/types/internal"

	"math"
)

// convertToInt dispatches to specific int conversion functions (int = int32 on 32-bit)
func convertToInt[U Number](dst []int, src []U, n int) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]

	switch s := any(src).(type) {
	case []int:
		copy(dst, s)
	case []int8:
		convertFromInt8ToInt(dst, s)
	case []int16:
		convertFromInt16ToInt(dst, s)
	case []int32:
		convertFromInt32ToInt(dst, s)
	case []float32:
		convertToIntFromFloat32(dst, s)
	case []complex64:
		convertToIntFromComplex64(dst, s)
	}
}

// convertToIntFromFloat32 converts float32 to int (int = int32 on 32-bit)
func convertToIntFromFloat32(dst []int, src []float32) {
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
			dst[i] = int(val)
		}
	}
}

// convertToIntFromComplex64 converts complex64 to int (int = int32 on 32-bit)
func convertToIntFromComplex64(dst []int, src []complex64) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := real(src[i]) * scaleFloat32ToInt32
		if val > float32(math.MaxInt32) {
			dst[i] = math.MaxInt32
		} else if val < float32(math.MinInt32) {
			dst[i] = math.MinInt32
		} else {
			dst[i] = int(val)
		}
	}
}

// convertFromInt8ToInt converts int8 to int (int = int32 on 32-bit)
func convertFromInt8ToInt(dst []int, src []int8) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int(src[i]) * 16777216 // Scale up by 2^24
	}
}

// convertFromInt16ToInt converts int16 to int (int = int32 on 32-bit)
func convertFromInt16ToInt(dst []int, src []int16) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int(src[i]) * 65536 // Scale up by 2^16
	}
}

// convertFromInt32ToInt converts int32 to int (int = int32 on 32-bit, just copy)
func convertFromInt32ToInt(dst []int, src []int32) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int(src[i])
	}
}

// convertToInt8FromInt converts int to int8 with dithering (int = int32 on 32-bit)
func convertToInt8FromInt(dst []int8, src []int) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(src[0]) + 0x200000000
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := float32(src[i])
		scaled := val / 16777216.0 // int32 -> int8
		dither := tpdfDither(rng)
		result := scaled + dither
		
		if result > 127.0 {
			dst[i] = 127
		} else if result < -128.0 {
			dst[i] = -128
		} else {
			dst[i] = int8(result)
		}
	}
}

// convertToInt16FromInt converts int to int16 with dithering (int = int32 on 32-bit)
func convertToInt16FromInt(dst []int16, src []int) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	seed := uint64(0x123456789ABCDEF0)
	if len(src) > 0 {
		seed = uint64(src[0]) + 0x400000000
	}
	rng := newXorshift64Star(seed)

	for i := range len(dst) {
		val := float32(src[i])
		scaled := val / 65536.0 // int32 -> int16
		dither := tpdfDither(rng)
		result := scaled + dither
		
		if result > 32767.0 {
			dst[i] = 32767
		} else if result < -32768.0 {
			dst[i] = -32768
		} else {
			dst[i] = int16(result)
		}
	}
}

// convertToInt32FromInt converts int to int32 (int = int32 on 32-bit, just copy)
func convertToInt32FromInt(dst []int32, src []int) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = int32(src[i])
	}
}

// convertFromIntToFloat32 converts int to float32 (int = int32 on 32-bit)
func convertFromIntToFloat32(dst []float32, src []int) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		dst[i] = float32(src[i]) * scaleInt32ToFloat32
	}
}

// convertFromIntToComplex64 converts int to complex64 (int = int32 on 32-bit)
func convertFromIntToComplex64(dst []complex64, src []int) {
	if len(dst) == 0 {
		return
	}
	dst = dst[:len(src)]
	
	for i := range len(dst) {
		val := float32(src[i]) * scaleInt32ToFloat32
		dst[i] = complex(val, 0)
	}
}

