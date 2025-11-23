package types

import (
	"math"
	"testing"
)

func TestValueConvert_RealToReal(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  any
	}{
		// float32 conversions
		{"float32 to float32", float32(3.14), float32(3.14)},
		{"float32 to float64", float32(3.14), float64(3.14)},
		{"float32 to int64", float32(3.14), int64(3)},
		{"float32 to int", float32(3.14), int(3)},
		{"float32 to int32", float32(3.14), int32(3)},
		{"float32 to int16", float32(3.14), int16(3)},
		{"float32 to int8", float32(3.14), int8(3)},

		// float64 conversions
		{"float64 to float32", float64(3.14), float32(3.14)},
		{"float64 to float64", float64(3.14), float64(3.14)},
		{"float64 to int64", float64(3.14), int64(3)},
		{"float64 to int", float64(3.14), int(3)},
		{"float64 to int32", float64(3.14), int32(3)},
		{"float64 to int16", float64(3.14), int16(3)},
		{"float64 to int8", float64(3.14), int8(3)},

		// int conversions
		{"int to float32", int(42), float32(42)},
		{"int to float64", int(42), float64(42)},
		{"int to int64", int(42), int64(42)},
		{"int to int32", int(42), int32(42)},
		{"int to int16", int(42), int16(42)},
		{"int to int8", int(42), int8(42)},

		// int64 conversions
		{"int64 to float32", int64(42), float32(42)},
		{"int64 to float64", int64(42), float64(42)},
		{"int64 to int", int64(42), int(42)},
		{"int64 to int32", int64(42), int32(42)},
		{"int64 to int16", int64(42), int16(42)},
		{"int64 to int8", int64(42), int8(42)},

		// Clamping tests
		{"float64 to int8 clamp high", float64(300), int8(127)},
		{"float64 to int8 clamp low", float64(-300), int8(-128)},
		{"float64 to int16 clamp high", float64(40000), int16(32767)},
		{"float64 to int16 clamp low", float64(-40000), int16(-32768)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch want := tt.want.(type) {
			case float32:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, float32](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, float32](v)
					if math.Abs(float64(got-want)) > 1e-6 {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int:
					got := ValueConvert[int, float32](v)
					if math.Abs(float64(got-want)) > 1e-6 {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, float32](v)
					if math.Abs(float64(got-want)) > 1e-6 {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				default:
					t.Fatalf("unexpected source type for float32 destination: %T", tt.value)
				}
			case float64:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, float64](v)
					if math.Abs(got-want) > 1e-6 {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, float64](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			case int64:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, int64](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, int64](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, int64](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int:
					got := ValueConvert[int, int64](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			case int:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, int](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, int](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, int](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int:
					got := ValueConvert[int, int](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			case int32:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, int32](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, int32](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, int32](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int:
					got := ValueConvert[int, int32](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int32:
					got := ValueConvert[int32, int32](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			case int16:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, int16](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, int16](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, int16](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int:
					got := ValueConvert[int, int16](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int32:
					got := ValueConvert[int32, int16](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int16:
					got := ValueConvert[int16, int16](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			case int8:
				switch v := tt.value.(type) {
				case float32:
					got := ValueConvert[float32, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case float64:
					got := ValueConvert[float64, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int64:
					got := ValueConvert[int64, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int:
					got := ValueConvert[int, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int32:
					got := ValueConvert[int32, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int16:
					got := ValueConvert[int16, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				case int8:
					got := ValueConvert[int8, int8](v)
					if got != want {
						t.Errorf("ValueConvert() = %v, want %v", got, want)
					}
				}
			}
		})
	}
}

func TestValueConvert_ComplexToComplex(t *testing.T) {
	tests := []struct {
		name  string
		value complex64
		want  complex64
	}{
		{"complex64 to complex64", complex64(complex(3.14, 2.71)), complex64(complex(3.14, 2.71))},
		{"complex64 to complex64 zero imag", complex64(complex(5.0, 0)), complex64(complex(5.0, 0))},
		{"complex64 to complex64 negative", complex64(complex(-3.14, -2.71)), complex64(complex(-3.14, -2.71))},
		{"complex64 to complex64 large real", complex64(complex(1e10, 1e9)), complex64(complex(1e10, 1e9))},
		{"complex64 to complex64 large imag", complex64(complex(1e9, 1e10)), complex64(complex(1e9, 1e10))},
		{"complex64 to complex64 negative both", complex64(complex(-1e10, -1e9)), complex64(complex(-1e10, -1e9))},
		{"complex64 to complex64 zero", complex64(0), complex64(0)},
		{"complex64 to complex64 small values", complex64(complex(1e-10, 1e-9)), complex64(complex(1e-10, 1e-9))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ValueConvert[complex64, complex64](tt.value)
			if real(got) != real(tt.want) || imag(got) != imag(tt.want) {
				t.Errorf("ValueConvert() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValueConvert_ComplexToReal(t *testing.T) {
	tests := []struct {
		name  string
		value complex64
		want  any
	}{
		// Basic conversions
		{"complex64 to float32", complex64(complex(3.14, 2.71)), float32(3.14)},
		{"complex64 to float64", complex64(complex(3.14, 2.71)), float64(3.14)},
		{"complex64 to int64", complex64(complex(3.14, 2.71)), int64(3)},
		{"complex64 to int", complex64(complex(3.14, 2.71)), int(3)},
		{"complex64 to int32", complex64(complex(3.14, 2.71)), int32(3)},
		{"complex64 to int16", complex64(complex(3.14, 2.71)), int16(3)},
		{"complex64 to int8", complex64(complex(3.14, 2.71)), int8(3)},

		// Negative values
		{"complex64 to float32 negative", complex64(complex(-3.14, -2.71)), float32(-3.14)},
		{"complex64 to float64 negative", complex64(complex(-3.14, -2.71)), float64(-3.14)},
		{"complex64 to int negative", complex64(complex(-3.14, -2.71)), int(-3)},
		{"complex64 to int32 negative", complex64(complex(-3.14, -2.71)), int32(-3)},
		{"complex64 to int16 negative", complex64(complex(-3.14, -2.71)), int16(-3)},
		{"complex64 to int8 negative", complex64(complex(-3.14, -2.71)), int8(-3)},

		// Clamping tests for int8
		{"complex64 to int8 clamp high", complex64(complex(300, 0)), int8(127)},
		{"complex64 to int8 clamp low", complex64(complex(-300, 0)), int8(-128)},
		{"complex64 to int8 clamp very high", complex64(complex(1e10, 0)), int8(127)},
		{"complex64 to int8 clamp very low", complex64(complex(-1e10, 0)), int8(-128)},

		// Clamping tests for int16
		{"complex64 to int16 clamp high", complex64(complex(50000, 0)), int16(32767)},
		{"complex64 to int16 clamp low", complex64(complex(-50000, 0)), int16(-32768)},
		{"complex64 to int16 clamp very high", complex64(complex(1e10, 0)), int16(32767)},
		{"complex64 to int16 clamp very low", complex64(complex(-1e10, 0)), int16(-32768)},

		// Clamping tests for int32
		{"complex64 to int32 clamp high", complex64(complex(3e9, 0)), int32(math.MaxInt32)},
		{"complex64 to int32 clamp low", complex64(complex(-3e9, 0)), int32(math.MinInt32)},
		{"complex64 to int32 clamp very high", complex64(complex(1e20, 0)), int32(math.MaxInt32)},
		{"complex64 to int32 clamp very low", complex64(complex(-1e20, 0)), int32(math.MinInt32)},

		// Clamping tests for int64
		{"complex64 to int64 clamp high", complex64(complex(1e20, 0)), int64(math.MaxInt64)},
		{"complex64 to int64 clamp low", complex64(complex(-1e20, 0)), int64(math.MinInt64)},

		// Clamping tests for int (platform-specific)
		{"complex64 to int clamp high", complex64(complex(1e20, 0)), int(math.MaxInt)},
		{"complex64 to int clamp low", complex64(complex(-1e20, 0)), int(math.MinInt)},

		// Edge cases
		{"complex64 zero to float32", complex64(0), float32(0)},
		{"complex64 zero to int", complex64(0), int(0)},
		{"complex64 zero to int8", complex64(0), int8(0)},
		{"complex64 with large imag part", complex64(complex(42, 1e10)), float32(42)},     // imag part ignored
		{"complex64 with negative imag part", complex64(complex(42, -1e10)), float32(42)}, // imag part ignored
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch want := tt.want.(type) {
			case float32:
				got := ValueConvert[complex64, float32](tt.value)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case float64:
				got := ValueConvert[complex64, float64](tt.value)
				if math.Abs(float64(got-want)) > 1e-6 {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int64:
				got := ValueConvert[complex64, int64](tt.value)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int:
				got := ValueConvert[complex64, int](tt.value)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int32:
				got := ValueConvert[complex64, int32](tt.value)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int16:
				got := ValueConvert[complex64, int16](tt.value)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int8:
				got := ValueConvert[complex64, int8](tt.value)
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			}
		})
	}
}

func TestValueConvert_RealToComplex(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  complex64
	}{
		// Basic conversions
		{"float32 to complex64", float32(3.14), complex64(complex(3.14, 0))},
		{"float64 to complex64", float64(3.14), complex64(complex(3.14, 0))},
		{"int to complex64", int(42), complex64(complex(42, 0))},
		{"int64 to complex64", int64(42), complex64(complex(42, 0))},
		{"int32 to complex64", int32(42), complex64(complex(42, 0))},
		{"int16 to complex64", int16(42), complex64(complex(42, 0))},
		{"int8 to complex64", int8(42), complex64(complex(42, 0))},

		// Negative values
		{"float32 negative to complex64", float32(-3.14), complex64(complex(-3.14, 0))},
		{"float64 negative to complex64", float64(-3.14), complex64(complex(-3.14, 0))},
		{"int negative to complex64", int(-42), complex64(complex(-42, 0))},
		{"int64 negative to complex64", int64(-42), complex64(complex(-42, 0))},
		{"int32 negative to complex64", int32(-42), complex64(complex(-42, 0))},
		{"int16 negative to complex64", int16(-42), complex64(complex(-42, 0))},
		{"int8 negative to complex64", int8(-42), complex64(complex(-42, 0))},

		// Edge cases
		{"float32 zero to complex64", float32(0), complex64(0)},
		{"float64 zero to complex64", float64(0), complex64(0)},
		{"int zero to complex64", int(0), complex64(0)},
		{"int8 zero to complex64", int8(0), complex64(0)},

		// Boundary values
		{"int8 max to complex64", int8(127), complex64(complex(127, 0))},
		{"int8 min to complex64", int8(-128), complex64(complex(-128, 0))},
		{"int16 max to complex64", int16(32767), complex64(complex(32767, 0))},
		{"int16 min to complex64", int16(-32768), complex64(complex(-32768, 0))},
		{"int32 max to complex64", int32(math.MaxInt32), complex64(complex(float32(math.MaxInt32), 0))},
		{"int32 min to complex64", int32(math.MinInt32), complex64(complex(float32(math.MinInt32), 0))},
		{"int64 max to complex64", int64(math.MaxInt64), complex64(complex(float32(math.MaxInt64), 0))},
		{"int64 min to complex64", int64(math.MinInt64), complex64(complex(float32(math.MinInt64), 0))},

		// Large float values
		{"float32 large to complex64", float32(1e10), complex64(complex(1e10, 0))},
		{"float64 large to complex64", float64(1e20), complex64(complex(1e20, 0))},
		{"float32 small to complex64", float32(1e-10), complex64(complex(1e-10, 0))},
		{"float64 small to complex64", float64(1e-20), complex64(complex(1e-20, 0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got complex64
			switch v := tt.value.(type) {
			case float32:
				got = ValueConvert[float32, complex64](v)
			case float64:
				got = ValueConvert[float64, complex64](v)
			case int:
				got = ValueConvert[int, complex64](v)
			case int64:
				got = ValueConvert[int64, complex64](v)
			case int32:
				got = ValueConvert[int32, complex64](v)
			case int16:
				got = ValueConvert[int16, complex64](v)
			case int8:
				got = ValueConvert[int8, complex64](v)
			}
			if real(got) != real(tt.want) || imag(got) != imag(tt.want) {
				t.Errorf("ValueConvert() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValueConvert_EdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  any
	}{
		// Zero values
		{"zero float32", float32(0), float32(0)},
		{"zero float64", float64(0), float64(0)},
		{"zero complex64", complex64(0), complex64(0)},
		{"zero int", int(0), int(0)},
		{"zero int8", int8(0), int8(0)},
		{"complex64 zero to int", complex64(0), int(0)},
		{"complex64 zero to float32", complex64(0), float32(0)},
		{"int zero to complex64", int(0), complex64(0)},

		// Boundary values - int8
		{"max int8 to complex64", int8(127), complex64(complex(127, 0))},
		{"min int8 to complex64", int8(-128), complex64(complex(-128, 0))},
		{"complex64 with large real to int8", complex64(complex(300, 0)), int8(127)},
		{"complex64 with small real to int8", complex64(complex(-300, 0)), int8(-128)},
		{"complex64 with max int8 value", complex64(complex(127, 0)), int8(127)},
		{"complex64 with min int8 value", complex64(complex(-128, 0)), int8(-128)},

		// Boundary values - int16
		{"max int16 to complex64", int16(32767), complex64(complex(32767, 0))},
		{"min int16 to complex64", int16(-32768), complex64(complex(-32768, 0))},
		{"complex64 with large real to int16", complex64(complex(50000, 0)), int16(32767)},
		{"complex64 with small real to int16", complex64(complex(-50000, 0)), int16(-32768)},
		{"complex64 with max int16 value", complex64(complex(32767, 0)), int16(32767)},
		{"complex64 with min int16 value", complex64(complex(-32768, 0)), int16(-32768)},

		// Boundary values - int32
		{"max int32 to complex64", int32(math.MaxInt32), complex64(complex(float32(math.MaxInt32), 0))},
		{"min int32 to complex64", int32(math.MinInt32), complex64(complex(float32(math.MinInt32), 0))},
		{"complex64 with large real to int32", complex64(complex(3e9, 0)), int32(math.MaxInt32)},
		{"complex64 with small real to int32", complex64(complex(-3e9, 0)), int32(math.MinInt32)},

		// Boundary values - int64
		{"max int64 to complex64", int64(math.MaxInt64), complex64(complex(float32(math.MaxInt64), 0))},
		{"min int64 to complex64", int64(math.MinInt64), complex64(complex(float32(math.MinInt64), 0))},
		{"complex64 with large real to int64", complex64(complex(1e20, 0)), int64(math.MaxInt64)},
		{"complex64 with small real to int64", complex64(complex(-1e20, 0)), int64(math.MinInt64)},

		// Complex with non-zero imaginary part (should be ignored)
		{"complex64 with imag to float32", complex64(complex(42, 100)), float32(42)},
		{"complex64 with imag to int", complex64(complex(42, 100)), int(42)},
		{"complex64 with imag to int8", complex64(complex(42, 100)), int8(42)},
		{"complex64 with large imag to float32", complex64(complex(42, 1e10)), float32(42)},
		{"complex64 with negative imag to float32", complex64(complex(42, -1e10)), float32(42)},

		// Precision edge cases
		{"float32 precision to complex64", float32(3.14159265), complex64(complex(3.14159265, 0))},
		{"float64 precision to complex64", float64(3.141592653589793), complex64(complex(3.141592653589793, 0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch want := tt.want.(type) {
			case float32:
				var got float32
				switch v := tt.value.(type) {
				case float32:
					got = ValueConvert[float32, float32](v)
				case complex64:
					got = ValueConvert[complex64, float32](v)
				default:
					t.Fatalf("unexpected source type for float32 destination: %T", tt.value)
				}
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case complex64:
				var got complex64
				switch v := tt.value.(type) {
				case float32:
					got = ValueConvert[float32, complex64](v)
				case float64:
					got = ValueConvert[float64, complex64](v)
				case int:
					got = ValueConvert[int, complex64](v)
				case int64:
					got = ValueConvert[int64, complex64](v)
				case int32:
					got = ValueConvert[int32, complex64](v)
				case int16:
					got = ValueConvert[int16, complex64](v)
				case int8:
					got = ValueConvert[int8, complex64](v)
				case complex64:
					got = ValueConvert[complex64, complex64](v)
				}
				if real(got) != real(want) || imag(got) != imag(want) {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int:
				var got int
				switch v := tt.value.(type) {
				case int:
					got = ValueConvert[int, int](v)
				case complex64:
					got = ValueConvert[complex64, int](v)
				default:
					t.Fatalf("unexpected source type for int destination: %T", tt.value)
				}
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			case int8:
				var got int8
				switch v := tt.value.(type) {
				case int8:
					got = ValueConvert[int8, int8](v)
				case complex64:
					got = ValueConvert[complex64, int8](v)
				default:
					t.Fatalf("unexpected source type for int8 destination: %T", tt.value)
				}
				if got != want {
					t.Errorf("ValueConvert() = %v, want %v", got, want)
				}
			}
		})
	}
}
