package st

import (
	"math"
	"testing"
)

func TestElemConvertClamping(t *testing.T) {
	tests := []struct {
		name    string
		src     any
		dstType string
		n       int
		want    any
	}{
		// int8 clamping tests
		{
			name:    "float64_to_int8_positive_clamp",
			src:     []float64{1000, 500, 127.5, 50, -50},
			dstType: "int8",
			n:       5,
			want:    []int8{127, 127, 127, 50, -50},
		},
		{
			name:    "float64_to_int8_negative_clamp",
			src:     []float64{-1000, -500, -128.5, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{-128, -128, -128, -50},
		},
		{
			name:    "float32_to_int8_clamp",
			src:     []float32{1000, -1000, 50, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		{
			name:    "int64_to_int8_clamp",
			src:     []int64{1000, -1000, 50, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		{
			name:    "int_to_int8_clamp",
			src:     []int{1000, -1000, 50, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		{
			name:    "int32_to_int8_clamp",
			src:     []int32{1000, -1000, 50, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		{
			name:    "int16_to_int8_clamp",
			src:     []int16{1000, -1000, 50, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		// int16 clamping tests
		{
			name:    "float64_to_int16_positive_clamp",
			src:     []float64{50000, 40000, 32767.5, 1000, -1000},
			dstType: "int16",
			n:       5,
			want:    []int16{32767, 32767, 32767, 1000, -1000},
		},
		{
			name:    "float64_to_int16_negative_clamp",
			src:     []float64{-50000, -40000, -32768.5, -1000},
			dstType: "int16",
			n:       4,
			want:    []int16{-32768, -32768, -32768, -1000},
		},
		{
			name:    "float32_to_int16_clamp",
			src:     []float32{50000, -50000, 1000, -1000},
			dstType: "int16",
			n:       4,
			want:    []int16{32767, -32768, 1000, -1000},
		},
		{
			name:    "int64_to_int16_clamp",
			src:     []int64{50000, -50000, 1000, -1000},
			dstType: "int16",
			n:       4,
			want:    []int16{32767, -32768, 1000, -1000},
		},
		{
			name:    "int_to_int16_clamp",
			src:     []int{50000, -50000, 1000, -1000},
			dstType: "int16",
			n:       4,
			want:    []int16{32767, -32768, 1000, -1000},
		},
		{
			name:    "int32_to_int16_clamp",
			src:     []int32{50000, -50000, 1000, -1000},
			dstType: "int16",
			n:       4,
			want:    []int16{32767, -32768, 1000, -1000},
		},
		// int32 clamping tests
		{
			name:    "float64_to_int32_positive_clamp",
			src:     []float64{3e9, 2.5e9, float64(math.MaxInt32) + 1, 1000, -1000},
			dstType: "int32",
			n:       5,
			want:    []int32{math.MaxInt32, math.MaxInt32, math.MaxInt32, 1000, -1000},
		},
		{
			name:    "float64_to_int32_negative_clamp",
			src:     []float64{-3e9, -2.5e9, float64(math.MinInt32) - 1, -1000},
			dstType: "int32",
			n:       4,
			want:    []int32{math.MinInt32, math.MinInt32, math.MinInt32, -1000},
		},
		{
			name:    "float32_to_int32_clamp",
			src:     []float32{3e9, -3e9, 1000, -1000},
			dstType: "int32",
			n:       4,
			want:    []int32{math.MaxInt32, math.MinInt32, 1000, -1000},
		},
		{
			name:    "int64_to_int32_clamp",
			src:     []int64{3000000000, -3000000000, 1000, -1000},
			dstType: "int32",
			n:       4,
			want:    []int32{math.MaxInt32, math.MinInt32, 1000, -1000},
		},
		{
			name:    "int_to_int32_clamp",
			src:     []int{math.MaxInt, math.MinInt, 1000, -1000},
			dstType: "int32",
			n:       4,
			want:    []int32{math.MaxInt32, math.MinInt32, 1000, -1000},
		},
		// int64 clamping tests
		{
			name:    "float64_to_int64_positive_clamp",
			src:     []float64{1e20, 1e19, float64(math.MaxInt64) + 1e10, 1000, -1000},
			dstType: "int64",
			n:       5,
			want:    []int64{math.MaxInt64, math.MaxInt64, math.MaxInt64, 1000, -1000},
		},
		{
			name:    "float64_to_int64_negative_clamp",
			src:     []float64{-1e20, -1e19, float64(math.MinInt64) - 1e10, -1000},
			dstType: "int64",
			n:       4,
			want:    []int64{math.MinInt64, math.MinInt64, math.MinInt64, -1000},
		},
		{
			name:    "float32_to_int64_clamp",
			src:     []float32{1e20, -1e20, 1000, -1000},
			dstType: "int64",
			n:       4,
			want:    []int64{math.MaxInt64, math.MinInt64, 1000, -1000},
		},
		// int clamping tests (platform-specific)
		{
			name:    "float64_to_int_clamp",
			src:     []float64{1e20, -1e20, 1000, -1000},
			dstType: "int",
			n:       4,
			want:    []int{math.MaxInt, math.MinInt, 1000, -1000},
		},
		{
			name:    "float32_to_int_clamp",
			src:     []float32{1e20, -1e20, 1000, -1000},
			dstType: "int",
			n:       4,
			want:    []int{math.MaxInt, math.MinInt, 1000, -1000},
		},
		{
			name:    "int64_to_int_clamp",
			src:     []int64{math.MaxInt64, math.MinInt64, 1000, -1000},
			dstType: "int",
			n:       4,
			want:    []int{math.MaxInt, math.MinInt, 1000, -1000},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.dstType {
			case "int8":
				dst := make([]int8, tt.n)
				switch s := tt.src.(type) {
				case []float64:
					ElemConvert(dst, s, tt.n)
				case []float32:
					ElemConvert(dst, s, tt.n)
				case []int64:
					ElemConvert(dst, s, tt.n)
				case []int:
					ElemConvert(dst, s, tt.n)
				case []int32:
					ElemConvert(dst, s, tt.n)
				case []int16:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int8)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int16":
				dst := make([]int16, tt.n)
				switch s := tt.src.(type) {
				case []float64:
					ElemConvert(dst, s, tt.n)
				case []float32:
					ElemConvert(dst, s, tt.n)
				case []int64:
					ElemConvert(dst, s, tt.n)
				case []int:
					ElemConvert(dst, s, tt.n)
				case []int32:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int16)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int32":
				dst := make([]int32, tt.n)
				switch s := tt.src.(type) {
				case []float64:
					ElemConvert(dst, s, tt.n)
				case []float32:
					ElemConvert(dst, s, tt.n)
				case []int64:
					ElemConvert(dst, s, tt.n)
				case []int:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int32)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int64":
				dst := make([]int64, tt.n)
				switch s := tt.src.(type) {
				case []float64:
					ElemConvert(dst, s, tt.n)
				case []float32:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int64)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int":
				dst := make([]int, tt.n)
				switch s := tt.src.(type) {
				case []float64:
					ElemConvert(dst, s, tt.n)
				case []float32:
					ElemConvert(dst, s, tt.n)
				case []int64:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			}
		})
	}
}

func TestElemConvertComplex64Clamping(t *testing.T) {
	tests := []struct {
		name    string
		src     []complex64
		dstType string
		n       int
		want    any
	}{
		{
			name:    "complex64_to_int8_clamp",
			src:     []complex64{complex(1000, 0), complex(-1000, 0), complex(50, 0), complex(-50, 0)},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		{
			name:    "complex64_to_int16_clamp",
			src:     []complex64{complex(50000, 0), complex(-50000, 0), complex(1000, 0), complex(-1000, 0)},
			dstType: "int16",
			n:       4,
			want:    []int16{32767, -32768, 1000, -1000},
		},
		{
			name:    "complex64_to_int32_clamp",
			src:     []complex64{complex(3e9, 0), complex(-3e9, 0), complex(1000, 0), complex(-1000, 0)},
			dstType: "int32",
			n:       4,
			want:    []int32{math.MaxInt32, math.MinInt32, 1000, -1000},
		},
		{
			name:    "complex64_to_int64_clamp",
			src:     []complex64{complex(1e20, 0), complex(-1e20, 0), complex(1000, 0), complex(-1000, 0)},
			dstType: "int64",
			n:       4,
			want:    []int64{math.MaxInt64, math.MinInt64, 1000, -1000},
		},
		{
			name:    "complex64_to_int_clamp",
			src:     []complex64{complex(1e20, 0), complex(-1e20, 0), complex(1000, 0), complex(-1000, 0)},
			dstType: "int",
			n:       4,
			want:    []int{math.MaxInt, math.MinInt, 1000, -1000},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.dstType {
			case "int8":
				dst := make([]int8, tt.n)
				ElemConvert(dst, tt.src, tt.n)
				want := tt.want.([]int8)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int16":
				dst := make([]int16, tt.n)
				ElemConvert(dst, tt.src, tt.n)
				want := tt.want.([]int16)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int32":
				dst := make([]int32, tt.n)
				ElemConvert(dst, tt.src, tt.n)
				want := tt.want.([]int32)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int64":
				dst := make([]int64, tt.n)
				ElemConvert(dst, tt.src, tt.n)
				want := tt.want.([]int64)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int":
				dst := make([]int, tt.n)
				ElemConvert(dst, tt.src, tt.n)
				want := tt.want.([]int)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			}
		})
	}
}

func TestElemConvertNoClamping(t *testing.T) {
	tests := []struct {
		name    string
		src     any
		dstType string
		n       int
		want    any
	}{
		// These conversions should NOT clamp (lower range to higher range)
		{
			name:    "int8_to_int16",
			src:     []int8{100, -100, 50, -50},
			dstType: "int16",
			n:       4,
			want:    []int16{100, -100, 50, -50},
		},
		{
			name:    "int8_to_int32",
			src:     []int8{100, -100, 50, -50},
			dstType: "int32",
			n:       4,
			want:    []int32{100, -100, 50, -50},
		},
		{
			name:    "int16_to_int32",
			src:     []int16{1000, -1000, 500, -500},
			dstType: "int32",
			n:       4,
			want:    []int32{1000, -1000, 500, -500},
		},
		{
			name:    "int32_to_int64",
			src:     []int32{100000, -100000, 50000, -50000},
			dstType: "int64",
			n:       4,
			want:    []int64{100000, -100000, 50000, -50000},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.dstType {
			case "int16":
				dst := make([]int16, tt.n)
				switch s := tt.src.(type) {
				case []int8:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int16)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int32":
				dst := make([]int32, tt.n)
				switch s := tt.src.(type) {
				case []int8:
					ElemConvert(dst, s, tt.n)
				case []int16:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int32)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			case "int64":
				dst := make([]int64, tt.n)
				switch s := tt.src.(type) {
				case []int32:
					ElemConvert(dst, s, tt.n)
				}
				want := tt.want.([]int64)
				for i := range want {
					if dst[i] != want[i] {
						t.Errorf("ElemConvert() dst[%d] = %v, want %v", i, dst[i], want[i])
					}
				}
			}
		})
	}
}
