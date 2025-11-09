package types

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
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
			expected: FP64,
		},
		{
			name:     "float32 scalar",
			data:     float32(1.5),
			expected: FP32,
		},
		{
			name:     "int16 scalar",
			data:     int16(42),
			expected: INT16,
		},
		{
			name:     "int8 scalar",
			data:     int8(42),
			expected: INT8,
		},
		{
			name:     "float64 slice",
			data:     []float64{1.0, 2.0, 3.0},
			expected: FP64,
		},
		{
			name:     "float32 slice",
			data:     []float32{1.0, 2.0, 3.0},
			expected: FP32,
		},
		{
			name:     "int16 slice",
			data:     []int16{1, 2, 3},
			expected: INT16,
		},
		{
			name:     "int8 slice",
			data:     []int8{1, 2, 3},
			expected: INT8,
		},
		{
			name:     "int32 scalar",
			data:     int32(42),
			expected: INT32,
		},
		{
			name:     "int32 slice",
			data:     []int32{1, 2, 3},
			expected: INT32,
		},
		{
			name:     "int64 scalar",
			data:     int64(42),
			expected: INT64,
		},
		{
			name:     "int64 slice",
			data:     []int64{1, 2, 3},
			expected: INT64,
		},
		{
			name:     "int scalar",
			data:     int(42),
			expected: INT,
		},
		{
			name:     "int slice",
			data:     []int{1, 2, 3},
			expected: INT,
		},
		{
			name:     "unknown type (string)",
			data:     "not a number",
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
			dt:       FP32,
			size:     5,
			expected: make([]float32, 5),
		},
		{
			name:     "DTFP64",
			dt:       FP64,
			size:     5,
			expected: make([]float64, 5),
		},
		{
			name:     "DTINT16",
			dt:       INT16,
			size:     5,
			expected: make([]int16, 5),
		},
		{
			name:     "DTINT8",
			dt:       INT8,
			size:     5,
			expected: make([]int8, 5),
		},
		{
			name:     "DTINT32",
			dt:       INT32,
			size:     5,
			expected: make([]int32, 5),
		},
		{
			name:     "DTINT64",
			dt:       INT64,
			size:     5,
			expected: make([]int64, 5),
		},
		{
			name:     "DTINT",
			dt:       INT,
			size:     5,
			expected: make([]int, 5),
		},
		{
			name:     "DTINT48",
			dt:       INT48,
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
			dt:       FP32,
			size:     0,
			expected: make([]float32, 0),
		},
		{
			name:     "large size",
			dt:       FP32,
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
				case []int32:
					actual, ok := result.([]int32)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
				case []int64:
					actual, ok := result.([]int64)
					assert.True(t, ok)
					assert.Equal(t, len(expected), len(actual))
				case []int:
					actual, ok := result.([]int)
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
			name:     "int32 slice",
			data:     []int32{1, 2, 3},
			expected: []int32{1, 2, 3},
		},
		{
			name:     "int64 slice",
			data:     []int64{1, 2, 3},
			expected: []int64{1, 2, 3},
		},
		{
			name:     "int slice",
			data:     []int{1, 2, 3},
			expected: []int{1, 2, 3},
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
			case []int32:
				originalData = append([]int32(nil), d...)
			case []int64:
				originalData = append([]int64(nil), d...)
			case []int:
				originalData = append([]int(nil), d...)
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

func TestReleaseTensorData(t *testing.T) {
	_ = fp32.Pool.Reconfigure()
	data := MakeTensorData(FP32, 8).([]float32)
	ptr := &data[0]

	ReleaseTensorData(data)

	reused := fp32.Pool.Get(8)
	if len(reused) > 0 {
		assert.Equal(t, ptr, &reused[0], "expected buffer to be returned to pool")
	}
	fp32.Pool.Put(reused)
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
			dst:      FP32,
			data:     []float32{1.0, 2.0, 3.0},
			expected: []float32{1.0, 2.0, 3.0},
		},
		{
			name:     "float64 to float32",
			dst:      FP32,
			data:     []float64{1.5, 2.5, 3.5},
			expected: []float32{1.5, 2.5, 3.5},
		},
		{
			name:     "float32 to float64",
			dst:      FP64,
			data:     []float32{1.5, 2.5, 3.5},
			expected: []float64{1.5, 2.5, 3.5},
		},
		{
			name:     "int16 to float32",
			dst:      FP32,
			data:     []int16{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
		},
		{
			name:     "int8 to float32",
			dst:      FP32,
			data:     []int8{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
		},
		{
			name:     "float32 to int16",
			dst:      INT16,
			data:     []float32{10.7, 20.3, 30.9},
			expected: []int16{10, 20, 30},
		},
		{
			name:     "float32 to int8",
			dst:      INT8,
			data:     []float32{10.7, 20.3, 30.9},
			expected: []int8{10, 20, 30},
		},
		{
			name:     "int16 to int8",
			dst:      INT8,
			data:     []int16{100, 200, 300},
			expected: []int8{100, 127, 127}, // clamped to int8 max
		},
		{
			name:     "int16 to int8 with negative clamping",
			dst:      INT8,
			data:     []int16{-100, -200, -300},
			expected: []int8{-100, -128, -128}, // clamped to int8 min
		},
		{
			name:     "int16 to int8 with both clamping",
			dst:      INT8,
			data:     []int16{-200, -100, 0, 100, 200},
			expected: []int8{-128, -100, 0, 100, 127}, // clamped to int8 range
		},
		{
			name:     "int8 to int16",
			dst:      INT16,
			data:     []int8{10, 20, 30},
			expected: []int16{10, 20, 30},
		},
		{
			name:     "int32 to float32",
			dst:      FP32,
			data:     []int32{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
		},
		{
			name:     "float32 to int32",
			dst:      INT32,
			data:     []float32{10.7, 20.3, 30.9},
			expected: []int32{10, 20, 30},
		},
		{
			name:     "int64 to float64",
			dst:      FP64,
			data:     []int64{10, 20, 30},
			expected: []float64{10.0, 20.0, 30.0},
		},
		{
			name:     "float64 to int64",
			dst:      INT64,
			data:     []float64{10.7, 20.3, 30.9},
			expected: []int64{10, 20, 30},
		},
		{
			name:     "int to float32",
			dst:      FP32,
			data:     []int{10, 20, 30},
			expected: []float32{10.0, 20.0, 30.0},
		},
		{
			name:     "float32 to int",
			dst:      INT,
			data:     []float32{10.7, 20.3, 30.9},
			expected: []int{10, 20, 30},
		},
		{
			name:     "int32 to int64",
			dst:      INT64,
			data:     []int32{10, 20, 30},
			expected: []int64{10, 20, 30},
		},
		{
			name:     "int64 to int32 with clamping",
			dst:      INT32,
			data:     []int64{3000000000, 2000000000, -3000000000},
			expected: []int32{2147483647, 2000000000, -2147483648},
		},
		{
			name:     "nil input",
			dst:      FP32,
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
				case []int32:
					actual, ok := result.([]int32)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int64:
					actual, ok := result.([]int64)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int:
					actual, ok := result.([]int)
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
			dstData:  []string{"a", "b", "c"}, // Wrong type (not numeric)
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
				case []int32:
					actual, ok := result.([]int32)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int64:
					actual, ok := result.([]int64)
					assert.True(t, ok)
					assert.Equal(t, expected, actual)
				case []int:
					actual, ok := result.([]int)
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
