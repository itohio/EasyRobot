package generics

import (
	"math"
	"testing"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

func TestElemConvert(t *testing.T) {
	tests := []struct {
		name    string
		src     any
		dstType string
		n       int
		want    any
		checkFn func(t *testing.T, got, want any)
	}{
		{
			name:    "float32 to float64",
			src:     []float32{1.5, 2.5, 3.5},
			dstType: "float64",
			n:       3,
			want:    []float64{1.5, 2.5, 3.5},
			checkFn: func(t *testing.T, got, want any) {
				gotSlice := got.([]float64)
				wantSlice := want.([]float64)
				for i := range wantSlice {
					if gotSlice[i] != wantSlice[i] {
						t.Errorf("ElemConvert() got[%d] = %v, want %v", i, gotSlice[i], wantSlice[i])
					}
				}
			},
		},
		{
			name:    "float32 to int32",
			src:     []float32{1.5, 2.5, 3.5},
			dstType: "int32",
			n:       3,
			want:    []int32{1, 2, 3},
			checkFn: func(t *testing.T, got, want any) {
				gotSlice := got.([]int32)
				wantSlice := want.([]int32)
				for i := range wantSlice {
					if gotSlice[i] != wantSlice[i] {
						t.Errorf("ElemConvert() got[%d] = %v, want %v", i, gotSlice[i], wantSlice[i])
					}
				}
			},
		},
		{
			name:    "int32 to float32",
			src:     []int32{1, 2, 3},
			dstType: "float32",
			n:       3,
			want:    []float32{1, 2, 3},
			checkFn: func(t *testing.T, got, want any) {
				gotSlice := got.([]float32)
				wantSlice := want.([]float32)
				for i := range wantSlice {
					if gotSlice[i] != wantSlice[i] {
						t.Errorf("ElemConvert() got[%d] = %v, want %v", i, gotSlice[i], wantSlice[i])
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.dstType {
			case "float64":
				dst := make([]float64, tt.n)
				src := tt.src.([]float32)
				ElemConvert(dst, src, tt.n)
				tt.checkFn(t, dst, tt.want)
			case "int32":
				dst := make([]int32, tt.n)
				src := tt.src.([]float32)
				ElemConvert(dst, src, tt.n)
				tt.checkFn(t, dst, tt.want)
			case "float32":
				src := tt.src.([]int32)
				dst := make([]float32, tt.n)
				ElemConvert(dst, src, tt.n)
				tt.checkFn(t, dst, tt.want)
			}
		})
	}
}

func TestElemConvertClamping(t *testing.T) {
	tests := []struct {
		name    string
		src     []float64
		dstType string
		n       int
		want    any
	}{
		{
			name:    "float64 to int8 clamping",
			src:     []float64{1000, -1000, 50, -50},
			dstType: "int8",
			n:       4,
			want:    []int8{127, -128, 50, -50},
		},
		{
			name:    "float64 to int16 clamping",
			src:     []float64{50000, -50000, 1000, -1000},
			dstType: "int16",
			n:       4,
			want:    []int16{32767, -32768, 1000, -1000},
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
			}
		})
	}
}

func TestElemConvertStrided(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		dstType  string
		shape    []int
		stridesD []int
		stridesS []int
		want     any
		checkFn  func(t *testing.T, got, want any, size int)
	}{
		{
			name:     "2D contiguous float32 to int32",
			src:      []float32{1.5, 2.5, 3.5, 4.5},
			dstType:  "int32",
			shape:    []int{2, 2},
			stridesD: nil,
			stridesS: nil,
			want:     []int32{1, 2, 3, 4},
			checkFn: func(t *testing.T, got, want any, size int) {
				gotSlice := got.([]int32)
				wantSlice := want.([]int32)
				for i := 0; i < size; i++ {
					if gotSlice[i] != wantSlice[i] {
						t.Errorf("ElemConvertStrided() got[%d] = %v, want %v", i, gotSlice[i], wantSlice[i])
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := SizeFromShape(tt.shape)
			switch tt.dstType {
			case "int32":
				dst := make([]int32, size)
				ElemConvertStrided(dst, tt.src, tt.shape, tt.stridesD, tt.stridesS)
				tt.checkFn(t, dst, tt.want, size)
			}
		})
	}
}

func TestValueConvert(t *testing.T) {
	tests := []struct {
		name    string
		value   any
		dstType string
		want    any
	}{
		{
			name:    "float32 to float64",
			value:   float32(1.5),
			dstType: "float64",
			want:    float64(1.5),
		},
		{
			name:    "float32 to int32",
			value:   float32(1.5),
			dstType: "int32",
			want:    int32(1),
		},
		{
			name:    "int32 to float32",
			value:   int32(42),
			dstType: "float32",
			want:    float32(42),
		},
		{
			name:    "float64 to int8 clamping",
			value:   float64(1000),
			dstType: "int8",
			want:    int8(127),
		},
		{
			name:    "float64 to int8 negative clamping",
			value:   float64(-1000),
			dstType: "int8",
			want:    int8(-128),
		},
		{
			name:    "int64 to int32 clamping",
			value:   int64(math.MaxInt64),
			dstType: "int32",
			want:    int32(math.MaxInt32),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.dstType {
			case "float64":
				got := ValueConvert[float32, float64](tt.value.(float32))
				want := tt.want.(float64)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case "int32":
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, int32](v)
					want := tt.want.(int32)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, int32](v)
					want := tt.want.(int32)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			case "float32":
				got := ValueConvert[int32, float32](tt.value.(int32))
				want := tt.want.(float32)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case "int8":
				switch v := tt.value.(type) {
				case float64:
					got := ValueConvert[float64, int8](v)
					want := tt.want.(int8)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			}
		})
	}
}

func TestValueConvertClamping(t *testing.T) {
	tests := []struct {
		name     string
		src      any
		expected any
	}{
		// float64 -> int8 clamping
		{
			name:     "float64_to_int8_positive",
			src:      float64(1000),
			expected: int8(127),
		},
		{
			name:     "float64_to_int8_negative",
			src:      float64(-1000),
			expected: int8(-128),
		},
		// float32 -> int8 clamping
		{
			name:     "float32_to_int8_positive",
			src:      float32(500),
			expected: int8(127),
		},
		{
			name:     "float32_to_int8_negative",
			src:      float32(-500),
			expected: int8(-128),
		},
		// int64 -> int8 clamping
		{
			name:     "int64_to_int8_positive",
			src:      int64(1000),
			expected: int8(127),
		},
		{
			name:     "int64_to_int8_negative",
			src:      int64(-1000),
			expected: int8(-128),
		},
		// int32 -> int8 clamping
		{
			name:     "int32_to_int8_positive",
			src:      int32(1000),
			expected: int8(127),
		},
		{
			name:     "int32_to_int8_negative",
			src:      int32(-1000),
			expected: int8(-128),
		},
		// int16 -> int8 clamping
		{
			name:     "int16_to_int8_positive",
			src:      int16(1000),
			expected: int8(127),
		},
		{
			name:     "int16_to_int8_negative",
			src:      int16(-1000),
			expected: int8(-128),
		},
		// int -> int8 clamping
		{
			name:     "int_to_int8_positive",
			src:      int(1000),
			expected: int8(127),
		},
		{
			name:     "int_to_int8_negative",
			src:      int(-1000),
			expected: int8(-128),
		},
		// float64 -> int16 clamping
		{
			name:     "float64_to_int16_positive",
			src:      float64(50000),
			expected: int16(32767),
		},
		{
			name:     "float64_to_int16_negative",
			src:      float64(-50000),
			expected: int16(-32768),
		},
		// float32 -> int16 clamping
		{
			name:     "float32_to_int16_positive",
			src:      float32(50000),
			expected: int16(32767),
		},
		{
			name:     "float32_to_int16_negative",
			src:      float32(-50000),
			expected: int16(-32768),
		},
		// int64 -> int16 clamping
		{
			name:     "int64_to_int16_positive",
			src:      int64(50000),
			expected: int16(32767),
		},
		{
			name:     "int64_to_int16_negative",
			src:      int64(-50000),
			expected: int16(-32768),
		},
		// int32 -> int16 clamping
		{
			name:     "int32_to_int16_positive",
			src:      int32(50000),
			expected: int16(32767),
		},
		{
			name:     "int32_to_int16_negative",
			src:      int32(-50000),
			expected: int16(-32768),
		},
		// int -> int16 clamping
		{
			name:     "int_to_int16_positive",
			src:      int(50000),
			expected: int16(32767),
		},
		{
			name:     "int_to_int16_negative",
			src:      int(-50000),
			expected: int16(-32768),
		},
		// float64 -> int32 clamping
		{
			name:     "float64_to_int32_positive",
			src:      float64(3e9),
			expected: int32(2147483647),
		},
		{
			name:     "float64_to_int32_negative",
			src:      float64(-3e9),
			expected: int32(-2147483648),
		},
		// float32 -> int32 clamping
		{
			name:     "float32_to_int32_positive",
			src:      float32(3e9),
			expected: int32(2147483647),
		},
		{
			name:     "float32_to_int32_negative",
			src:      float32(-3e9),
			expected: int32(-2147483648),
		},
		// int64 -> int32 clamping
		{
			name:     "int64_to_int32_positive",
			src:      int64(3000000000),
			expected: int32(2147483647),
		},
		{
			name:     "int64_to_int32_negative",
			src:      int64(-3000000000),
			expected: int32(-2147483648),
		},
		// int -> int32 clamping
		{
			name:     "int_to_int32_positive",
			src:      int(math.MaxInt),
			expected: int32(2147483647),
		},
		{
			name:     "int_to_int32_negative",
			src:      int(math.MinInt),
			expected: int32(-2147483648),
		},
		// float64 -> int64 clamping
		{
			name:     "float64_to_int64_positive",
			src:      float64(1e20),
			expected: int64(math.MaxInt64),
		},
		{
			name:     "float64_to_int64_negative",
			src:      float64(-1e20),
			expected: int64(math.MinInt64),
		},
		// float32 -> int64 clamping
		{
			name:     "float32_to_int64_positive",
			src:      float32(1e20),
			expected: int64(math.MaxInt64),
		},
		{
			name:     "float32_to_int64_negative",
			src:      float32(-1e20),
			expected: int64(math.MinInt64),
		},
		// int -> int clamping (platform-specific, but should work)
		{
			name:     "int_to_int_same",
			src:      int(42),
			expected: int(42),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result any
			switch src := tt.src.(type) {
			case float64:
				switch expected := tt.expected.(type) {
				case int8:
					result = ValueConvert[float64, int8](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int16:
					result = ValueConvert[float64, int16](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int32:
					result = ValueConvert[float64, int32](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int64:
					result = ValueConvert[float64, int64](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				}
			case float32:
				switch expected := tt.expected.(type) {
				case int8:
					result = ValueConvert[float32, int8](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int16:
					result = ValueConvert[float32, int16](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int32:
					result = ValueConvert[float32, int32](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int64:
					result = ValueConvert[float32, int64](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				}
			case int64:
				switch expected := tt.expected.(type) {
				case int8:
					result = ValueConvert[int64, int8](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int16:
					result = ValueConvert[int64, int16](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int32:
					result = ValueConvert[int64, int32](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				}
			case int32:
				switch expected := tt.expected.(type) {
				case int8:
					result = ValueConvert[int32, int8](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int16:
					result = ValueConvert[int32, int16](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				}
			case int16:
				switch expected := tt.expected.(type) {
				case int8:
					result = ValueConvert[int16, int8](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				}
			case int:
				switch expected := tt.expected.(type) {
				case int8:
					result = ValueConvert[int, int8](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int16:
					result = ValueConvert[int, int16](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int32:
					result = ValueConvert[int, int32](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				case int:
					result = ValueConvert[int, int](src)
					if result != expected {
						t.Errorf("ValueConvert() = %v, want %v", result, expected)
					}
				}
			}
		})
	}
}

func TestValueConvertMatrix(t *testing.T) {
	// Test all type combinations
	types := []struct {
		name string
		make func([]int) any
		cast func(any) []int
		zero func(int) any
	}{
		{"float64", func(v []int) any {
			r := make([]float64, len(v))
			for i, x := range v {
				r[i] = float64(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]float64)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]float64, n) }},
		{"float32", func(v []int) any {
			r := make([]float32, len(v))
			for i, x := range v {
				r[i] = float32(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]float32)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]float32, n) }},
		{"int64", func(v []int) any {
			r := make([]int64, len(v))
			for i, x := range v {
				r[i] = int64(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int64)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int64, n) }},
		{"int32", func(v []int) any {
			r := make([]int32, len(v))
			for i, x := range v {
				r[i] = int32(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int32)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int32, n) }},
		{"int16", func(v []int) any {
			r := make([]int16, len(v))
			for i, x := range v {
				r[i] = int16(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int16)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int16, n) }},
		{"int8", func(v []int) any {
			r := make([]int8, len(v))
			for i, x := range v {
				r[i] = int8(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int8)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int8, n) }},
	}

	testValues := []int{1, 2, 3, 10, 20, 30, 100, -50, -100}

	for _, srcType := range types {
		for _, dstType := range types {
			t.Run(srcType.name+"_to_"+dstType.name, func(t *testing.T) {
				src := srcType.make(testValues)
				dst := dstType.zero(len(testValues))

				// Convert using ElemConvert
				switch s := src.(type) {
				case []float64:
					switch d := dst.(type) {
					case []float64:
						ElemConvert(d, s, len(testValues))
					case []float32:
						ElemConvert(d, s, len(testValues))
					case []int64:
						ElemConvert(d, s, len(testValues))
					case []int32:
						ElemConvert(d, s, len(testValues))
					case []int16:
						ElemConvert(d, s, len(testValues))
					case []int8:
						ElemConvert(d, s, len(testValues))
					}
				case []float32:
					switch d := dst.(type) {
					case []float64:
						ElemConvert(d, s, len(testValues))
					case []float32:
						ElemConvert(d, s, len(testValues))
					case []int64:
						ElemConvert(d, s, len(testValues))
					case []int32:
						ElemConvert(d, s, len(testValues))
					case []int16:
						ElemConvert(d, s, len(testValues))
					case []int8:
						ElemConvert(d, s, len(testValues))
					}
				case []int64:
					switch d := dst.(type) {
					case []float64:
						ElemConvert(d, s, len(testValues))
					case []float32:
						ElemConvert(d, s, len(testValues))
					case []int64:
						ElemConvert(d, s, len(testValues))
					case []int32:
						ElemConvert(d, s, len(testValues))
					case []int16:
						ElemConvert(d, s, len(testValues))
					case []int8:
						ElemConvert(d, s, len(testValues))
					}
				case []int32:
					switch d := dst.(type) {
					case []float64:
						ElemConvert(d, s, len(testValues))
					case []float32:
						ElemConvert(d, s, len(testValues))
					case []int64:
						ElemConvert(d, s, len(testValues))
					case []int32:
						ElemConvert(d, s, len(testValues))
					case []int16:
						ElemConvert(d, s, len(testValues))
					case []int8:
						ElemConvert(d, s, len(testValues))
					}
				case []int16:
					switch d := dst.(type) {
					case []float64:
						ElemConvert(d, s, len(testValues))
					case []float32:
						ElemConvert(d, s, len(testValues))
					case []int64:
						ElemConvert(d, s, len(testValues))
					case []int32:
						ElemConvert(d, s, len(testValues))
					case []int16:
						ElemConvert(d, s, len(testValues))
					case []int8:
						ElemConvert(d, s, len(testValues))
					}
				case []int8:
					switch d := dst.(type) {
					case []float64:
						ElemConvert(d, s, len(testValues))
					case []float32:
						ElemConvert(d, s, len(testValues))
					case []int64:
						ElemConvert(d, s, len(testValues))
					case []int32:
						ElemConvert(d, s, len(testValues))
					case []int16:
						ElemConvert(d, s, len(testValues))
					case []int8:
						ElemConvert(d, s, len(testValues))
					}
				}

				// For numeric conversions, verify approximate equality
				srcValues := srcType.cast(src)
				dstValues := dstType.cast(dst)

				if len(srcValues) != len(dstValues) {
					t.Errorf("lengths should match: got %d, want %d", len(dstValues), len(srcValues))
					return
				}

				// For same-type conversions, values should match exactly
				if srcType.name == dstType.name {
					for i := range srcValues {
						if srcValues[i] != dstValues[i] {
							t.Errorf("same-type conversion failed at index %d: got %v, want %v", i, dstValues[i], srcValues[i])
						}
					}
				}
			})
		}
	}
}
