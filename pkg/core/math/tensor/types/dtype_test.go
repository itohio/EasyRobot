package types

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/stretchr/testify/assert"
)

func TestTypeFromData(t *testing.T) {
	tests := []struct {
		name     string
		data     any
		expected DataType
	}{
		{
			name:     "float64 scalar",
			data:     float64(1.5),
			expected: DTFP64,
		},
		{
			name:     "float32 scalar",
			data:     float32(1.5),
			expected: DTFP32,
		},
		{
			name:     "int16 scalar",
			data:     int16(42),
			expected: DTINT16,
		},
		{
			name:     "int8 scalar",
			data:     int8(42),
			expected: DTINT8,
		},
		{
			name:     "float64 slice",
			data:     []float64{1.0, 2.0, 3.0},
			expected: DTFP64,
		},
		{
			name:     "float32 slice",
			data:     []float32{1.0, 2.0, 3.0},
			expected: DTFP32,
		},
		{
			name:     "int16 slice",
			data:     []int16{1, 2, 3},
			expected: DTINT16,
		},
		{
			name:     "int8 slice",
			data:     []int8{1, 2, 3},
			expected: DTINT8,
		},
		{
			name:     "unknown type (string)",
			data:     "not a number",
			expected: DT_UNKNOWN,
		},
		{
			name:     "unknown type (int)",
			data:     int(42),
			expected: DT_UNKNOWN,
		},
		{
			name:     "unknown type (nil)",
			data:     nil,
			expected: DT_UNKNOWN,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := TypeFromData(tt.data)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMakeTensorData(t *testing.T) {
	tests := []struct {
		name     string
		dt       DataType
		size     int
		expected any
	}{
		{
			name:     "DTFP32",
			dt:       DTFP32,
			size:     5,
			expected: make([]float32, 5),
		},
		{
			name:     "DTFP64",
			dt:       DTFP64,
			size:     5,
			expected: make([]float64, 5),
		},
		{
			name:     "DTINT16",
			dt:       DTINT16,
			size:     5,
			expected: make([]int16, 5),
		},
		{
			name:     "DTINT8",
			dt:       DTINT8,
			size:     5,
			expected: make([]int8, 5),
		},
		{
			name:     "DTINT48",
			dt:       DTINT48,
			size:     5,
			expected: make([]int8, 5),
		},
		{
			name:     "DT_UNKNOWN",
			dt:       DT_UNKNOWN,
			size:     5,
			expected: nil,
		},
		{
			name:     "zero size",
			dt:       DTFP32,
			size:     0,
			expected: make([]float32, 0),
		},
		{
			name:     "large size",
			dt:       DTFP32,
			size:     1000,
			expected: make([]float32, 1000),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MakeTensorData(tt.dt, tt.size)
			if tt.expected == nil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				// Verify type and length
				switch expected := tt.expected.(type) {
				case []float32:
					actual, ok := result.([]float32)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
				case []float64:
					actual, ok := result.([]float64)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
				case []int16:
					actual, ok := result.([]int16)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
				case []int8:
					actual, ok := result.([]int8)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
				}
			}
		})
	}
}

func TestCloneTensorData(t *testing.T) {
	tests := []struct {
		name     string
		data     any
		expected any
	}{
		{
			name:     "float32 slice",
			data:     []float32{1.0, 2.0, 3.0},
			expected: []float32{1.0, 2.0, 3.0},
		},
		{
			name:     "float64 slice",
			data:     []float64{1.0, 2.0, 3.0},
			expected: []float64{1.0, 2.0, 3.0},
		},
		{
			name:     "int16 slice",
			data:     []int16{1, 2, 3},
			expected: []int16{1, 2, 3},
		},
		{
			name:     "int8 slice",
			data:     []int8{1, 2, 3},
			expected: []int8{1, 2, 3},
		},
		{
			name:     "nil input",
			data:     nil,
			expected: nil,
		},
		{
			name:     "empty slice",
			data:     []float32{},
			expected: []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a copy of data for modification test
			var originalData any
			switch d := tt.data.(type) {
			case []float32:
				originalData = append([]float32(nil), d...)
			case []float64:
				originalData = append([]float64(nil), d...)
			case []int16:
				originalData = append([]int16(nil), d...)
			case []int8:
				originalData = append([]int8(nil), d...)
			default:
				originalData = tt.data
			}

			result := CloneTensorData(tt.data)
			if tt.expected == nil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				// Verify values match
				switch expected := tt.expected.(type) {
				case []float32:
					actual, ok := result.([]float32)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
					// Verify it's a copy (modifying original shouldn't affect clone)
					if len(tt.data.([]float32)) > 0 {
						original := originalData.([]float32)[0]
						tt.data.([]float32)[0] = 999
						assert.NotEqual(t, 999, actual[0])
						assert.Equal(t, original, actual[0])
					}
				case []float64:
					actual, ok := result.([]float64)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int16:
					actual, ok := result.([]int16)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int8:
					actual, ok := result.([]int8)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				}
			}
		})
	}
}

func TestCloneTensorDataTo(t *testing.T) {
	tests := []struct {
		name     string
		dst      DataType
		data     any
		expected any
	}{
		{
			name:     "float32 to float32",
			dst:      DTFP32,
			data:     []float32{1.0, 2.0, 3.0},
			expected: []float32{1.0, 2.0, 3.0},
		},
		{
			name:     "float64 to float32",
			dst:      DTFP32,
			data:     []float64{1.5, 2.5, 3.5},
			expected: []float32{1.5, 2.5, 3.5},
		},
		{
			name:     "float32 to float64",
			dst:      DTFP64,
			data:     []float32{1.5, 2.5, 3.5},
			expected: []float64{1.5, 2.5, 3.5},
		},
		{
			name:     "int16 to float32",
			dst:      DTFP32,
			data:     []int16{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
		},
		{
			name:     "int8 to float32",
			dst:      DTFP32,
			data:     []int8{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
		},
		{
			name:     "float32 to int16",
			dst:      DTINT16,
			data:     []float32{10.7, 20.3, 30.9},
			expected: []int16{10, 20, 30},
		},
		{
			name:     "float32 to int8",
			dst:      DTINT8,
			data:     []float32{10.7, 20.3, 30.9},
			expected: []int8{10, 20, 30},
		},
		{
			name:     "int16 to int8",
			dst:      DTINT8,
			data:     []int16{100, 200, 300},
			expected: []int8{100, 127, 127}, // clamped to int8 max
		},
		{
			name:     "int16 to int8 with negative clamping",
			dst:      DTINT8,
			data:     []int16{-100, -200, -300},
			expected: []int8{-100, -128, -128}, // clamped to int8 min
		},
		{
			name:     "int16 to int8 with both clamping",
			dst:      DTINT8,
			data:     []int16{-200, -100, 0, 100, 200},
			expected: []int8{-128, -100, 0, 100, 127}, // clamped to int8 range
		},
		{
			name:     "int8 to int16",
			dst:      DTINT16,
			data:     []int8{10, 20, 30},
			expected: []int16{10, 20, 30},
		},
		{
			name:     "nil input",
			dst:      DTFP32,
			data:     nil,
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CloneTensorDataTo(tt.dst, tt.data)
			if tt.expected == nil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				// Verify values match (with type conversion)
				switch expected := tt.expected.(type) {
				case []float32:
					actual, ok := result.([]float32)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
					for i := range expected {
						assert.InDelta(t, float64(expected[i]), float64(actual[i]), 0.01)
					}
				case []float64:
					actual, ok := result.([]float64)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
					for i := range expected {
						assert.InDelta(t, expected[i], actual[i], 0.01)
					}
				case []int16:
					actual, ok := result.([]int16)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int8:
					actual, ok := result.([]int8)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				}
			}
		})
	}
}

// TestCopyTensorData is now testing primitive.CopyWithConversion
// This test is kept for backward compatibility but now uses the primitive function.
func TestCopyTensorData(t *testing.T) {
	tests := []struct {
		name     string
		dstData  any
		srcData  any
		expected any
		wantNil  bool
	}{
		{
			name:     "float32 to float32",
			dstData:  make([]float32, 3),
			srcData:  []float32{1.0, 2.0, 3.0},
			expected: []float32{1.0, 2.0, 3.0},
			wantNil:  false,
		},
		{
			name:     "float64 to float32",
			dstData:  make([]float32, 3),
			srcData:  []float64{1.5, 2.5, 3.5},
			expected: []float32{1.5, 2.5, 3.5},
			wantNil:  false,
		},
		{
			name:     "float32 to float64",
			dstData:  make([]float64, 3),
			srcData:  []float32{1.5, 2.5, 3.5},
			expected: []float64{1.5, 2.5, 3.5},
			wantNil:  false,
		},
		{
			name:     "int16 to float32",
			dstData:  make([]float32, 3),
			srcData:  []int16{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
			wantNil:  false,
		},
		{
			name:     "int8 to float32",
			dstData:  make([]float32, 3),
			srcData:  []int8{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
			wantNil:  false,
		},
		{
			name:     "float32 to int16",
			dstData:  make([]int16, 3),
			srcData:  []float32{10.7, 20.3, 30.9},
			expected: []int16{10, 20, 30},
			wantNil:  false,
		},
		{
			name:     "float32 to int8",
			dstData:  make([]int8, 3),
			srcData:  []float32{10.7, 20.3, 30.9},
			expected: []int8{10, 20, 30},
			wantNil:  false,
		},
		{
			name:     "int16 to int8",
			dstData:  make([]int8, 3),
			srcData:  []int16{100, 200, 300},
			expected: []int8{100, 127, 127}, // clamped to int8 max
			wantNil:  false,
		},
		{
			name:     "int16 to int8 with negative clamping",
			dstData:  make([]int8, 3),
			srcData:  []int16{-100, -200, -300},
			expected: []int8{-100, -128, -128}, // clamped to int8 min
			wantNil:  false,
		},
		{
			name:     "int16 to int8 with both clamping",
			dstData:  make([]int8, 5),
			srcData:  []int16{-200, -100, 0, 100, 200},
			expected: []int8{-128, -100, 0, 100, 127}, // clamped to int8 range
			wantNil:  false,
		},
		{
			name:     "int8 to int16",
			dstData:  make([]int16, 3),
			srcData:  []int8{10, 20, 30},
			expected: []int16{10, 20, 30},
			wantNil:  false,
		},
		{
			name:     "nil dst",
			dstData:  nil,
			srcData:  []float32{1.0, 2.0, 3.0},
			expected: nil,
			wantNil:  true,
		},
		{
			name:     "nil src",
			dstData:  make([]float32, 3),
			srcData:  nil,
			expected: nil,
			wantNil:  true,
		},
		{
			name:     "wrong dst type",
			dstData:  []int{1, 2, 3}, // Wrong type
			srcData:  []float32{1.0, 2.0, 3.0},
			expected: nil,
			wantNil:  true,
		},
		{
			name:     "unknown src type",
			dstData:  make([]float32, 3),
			srcData:  []string{"a", "b", "c"}, // Unknown type
			expected: nil,
			wantNil:  true,
		},
		{
			name:     "float32 to int8 (DTINT48 equivalent)",
			dstData:  make([]int8, 3),
			srcData:  []float32{10.7, 20.3, 30.9},
			expected: []int8{10, 20, 30},
			wantNil:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := primitive.CopyWithConversion(tt.dstData, tt.srcData)
			if tt.wantNil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				// Verify values match (with type conversion)
				switch expected := tt.expected.(type) {
				case []float32:
					actual, ok := result.([]float32)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
					for i := range expected {
						assert.InDelta(t, float64(expected[i]), float64(actual[i]), 0.01)
					}
				case []float64:
					actual, ok := result.([]float64)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
					for i := range expected {
						assert.InDelta(t, expected[i], actual[i], 0.01)
					}
				case []int16:
					actual, ok := result.([]int16)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int8:
					actual, ok := result.([]int8)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				}
			}
		})
	}
}

func TestGetTensorData(t *testing.T) {
	// Note: This function requires a Tensor interface, which we can't easily create
	// without importing the eager_tensor package. For now, we'll test the nil cases.
	t.Run("nil tensor", func(t *testing.T) {
		var result []float32
		result = GetTensorData[[]float32](nil)
		assert.Nil(t, result)
	})

	// The actual tensor implementation would need to be tested with a real tensor
	// This would require integration with eager_tensor package or a mock
}
