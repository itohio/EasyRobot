package types

import (
	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
)

// DataType represents the underlying element type stored by a tensor.
type DataType uint8

const (
	DT_UNKNOWN DataType = iota
	DTFP32              // DTFP32 represents 32-bit floating point tensors (default).
	DTFP64              // 64-bit floating point tensors
	DTFP16              // 16-bit floating point tensors
	DTINT48             // 4-bit integer tensors unpacked into 8bit
	DTINT8              // 8-bit integer tensors
	DTINT16             // 16-bit integer tensors
	DTINT               // 32-bit integer tensors
	DTINT32             // 32-bit integer tensors
	DTINT64             // 64-bit integer tensors
)

// DataElementType is the type constraint for the data elements in the tensor.
type DataElementType interface {
	~float64 | ~float32 | ~int16 | ~int8
}

func TypeFromData(v any) DataType {
	switch any(v).(type) {
	case float64:
		return DTFP64
	case float32:
		return DTFP32
	case int:
		return DTINT
	case int64:
		return DTINT64
	case int32:
		return DTINT32
	case int16:
		return DTINT16
	case int8:
		return DTINT8
	case []float64:
		return DTFP64
	case []float32:
		return DTFP32
	case []int16:
		return DTINT16
	case []int:
		return DTINT
	case []int32:
		return DTINT32
	case []int64:
		return DTINT64
	case []int8:
		return DTINT8
	default:
		return DT_UNKNOWN
	}
}

func MakeTensorData(dt DataType, size int) any {
	switch dt {
	case DTFP32:
		return make([]float32, size)
	case DTFP64:
		return make([]float64, size)
	case DTINT16:
		return make([]int16, size)
	case DTINT:
		return make([]int, size)
	case DTINT32:
		return make([]int32, size)
	case DTINT64:
		return make([]int64, size)
	case DTINT8:
		return make([]int8, size)
	case DTINT48:
		return make([]int8, size)
	default:
		return nil
	}
}

func CloneTensorDataTo(dst DataType, data any) any {
	if data == nil {
		return nil
	}
	size := 0
	switch d := data.(type) {
	case []float32:
		size = len(d)
	case []float64:
		size = len(d)
	case []int16:
		size = len(d)
	case []int:
		size = len(d)
	case []int32:
		size = len(d)
	case []int64:
		size = len(d)
	case []int8:
		size = len(d)
	default:
		return nil
	}

	newData := MakeTensorData(dst, size)
	if newData == nil {
		return nil
	}
	// Use primitive.CopyWithConversion instead of CopyTensorData
	primitive.CopyWithConversion(newData, data)
	return newData
}

func CloneTensorData(data any) any {
	if data == nil {
		return nil
	}
	return CloneTensorDataTo(TypeFromData(data), data)
}

// Helper functions to work with interface tensors
func GetTensorData[T any](t Tensor) T {
	if t == nil || t.Empty() {
		var zero T
		return zero
	}
	data := t.Data()
	if data == nil {
		var zero T
		return zero
	}
	return data.(T)
}
