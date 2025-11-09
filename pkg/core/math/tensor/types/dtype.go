package types

import (
	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/qi"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/qi16"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/qi32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/qi64"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/qi8"
)

// DataType represents the underlying element type stored by a tensor.
type DataType uint8

const (
	DT_UNKNOWN DataType = iota
	INT64               // 64-bit integer tensors
	FP64                // 64-bit floating point tensors
	INT32               // 32-bit integer tensors
	FP32                // DTFP32 represents 32-bit floating point tensors (default)
	INT                 // native integer tensors 32bit or 64bit
	INT16               // 16-bit integer tensors
	FP16                // 16-bit floating point tensors
	INT8                // 8-bit integer tensors
	INT48               // 4-bit integer tensors unpacked into 8bit
)

// DataElementType is the type constraint for the data elements in the tensor.
type DataElementType interface {
	~float64 | ~float32 | ~int64 | ~int | ~int32 | ~int16 | ~int8
}

func TypeFromData(v any) DataType {
	switch any(v).(type) {
	case float64:
		return FP64
	case float32:
		return FP32
	case int:
		return INT
	case int64:
		return INT64
	case int32:
		return INT32
	case int16:
		return INT16
	case int8:
		return INT8
	case []float64:
		return FP64
	case []float32:
		return FP32
	case []int16:
		return INT16
	case []int:
		return INT
	case []int32:
		return INT32
	case []int64:
		return INT64
	case []int8:
		return INT8
	default:
		return DT_UNKNOWN
	}
}

func MakeTensorData(dt DataType, size int) any {
	switch dt {
	case FP32:
		return fp32.Pool.Get(size)
	case FP64:
		return make([]float64, size)
	case INT16:
		return qi16.Pool.Get(size)
	case INT:
		return qi.Pool.Get(size)
	case INT32:
		return qi32.Pool.Get(size)
	case INT64:
		return qi64.Pool.Get(size)
	case INT8:
		return qi8.Pool.Get(size)
	case INT48:
		return qi8.Pool.Get(size)
	default:
		return nil
	}
}

func ReleaseTensorData(data any) {
	if data == nil {
		return
	}

	switch buf := data.(type) {
	case []float32:
		fp32.Pool.Put(buf)
	case []int16:
		qi16.Pool.Put(buf)
	case []int:
		qi.Pool.Put(buf)
	case []int32:
		qi32.Pool.Put(buf)
	case []int64:
		qi64.Pool.Put(buf)
	case []int8:
		qi8.Pool.Put(buf)
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

// Helper functions to work with tensors.
// Important: Passing in Tensor struct by value is expected.
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
