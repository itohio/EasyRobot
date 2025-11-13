package gob

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Conversion from gob structs back to domain types

func gobToTensor(gt gobTensor, tensorFactory func(types.DataType, types.Shape) types.Tensor) types.Tensor {
	if len(gt.Shape) == 0 {
		return nil
	}

	dtype := types.DataType(gt.DataType)
	shape := types.Shape(gt.Shape)

	// Use provided tensor factory or default to tensor.New wrapped
	var t types.Tensor
	if tensorFactory != nil {
		t = tensorFactory(dtype, shape)
	} else {
		t = tensor.New(dtype, shape)
	}

	// Copy data from gob struct to tensor
	// The data is already in the correct type from gob decoding
	switch dtype {
	case types.FP32:
		if data, ok := gt.Data.([]float32); ok {
			if dst, ok := t.Data().([]float32); ok {
				copy(dst, data)
			}
		}
	case types.FP64:
		if data, ok := gt.Data.([]float64); ok {
			if dst, ok := t.Data().([]float64); ok {
				copy(dst, data)
			}
		}
	case types.INT8:
		if data, ok := gt.Data.([]int8); ok {
			if dst, ok := t.Data().([]int8); ok {
				copy(dst, data)
			}
		}
	case types.INT16:
		if data, ok := gt.Data.([]int16); ok {
			if dst, ok := t.Data().([]int16); ok {
				copy(dst, data)
			}
		}
	case types.INT32:
		if data, ok := gt.Data.([]int32); ok {
			if dst, ok := t.Data().([]int32); ok {
				copy(dst, data)
			}
		}
	case types.INT64:
		if data, ok := gt.Data.([]int64); ok {
			if dst, ok := t.Data().([]int64); ok {
				copy(dst, data)
			}
		}
	case types.INT:
		if data, ok := gt.Data.([]int); ok {
			if dst, ok := t.Data().([]int); ok {
				copy(dst, data)
			}
		}
	case types.UINT8:
		if data, ok := gt.Data.([]uint8); ok {
			if dst, ok := t.Data().([]uint8); ok {
				copy(dst, data)
			}
		}
	}

	return t
}

func gobToTensorWithConversion(gt gobTensor, destType types.DataType, tensorFactory func(types.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if len(gt.Shape) == 0 {
		return nil, fmt.Errorf("empty tensor shape")
	}

	// First create the tensor with original type
	srcTensor := gobToTensor(gt, tensorFactory)
	if srcTensor == nil || srcTensor.Empty() {
		return nil, fmt.Errorf("failed to create source tensor")
	}

	// If no conversion needed, return as-is
	srcType := types.DataType(gt.DataType)
	if destType == 0 || destType == srcType {
		return srcTensor, nil
	}

	// Create destination tensor
	shape := types.Shape(gt.Shape)
	var destTensor types.Tensor
	if tensorFactory != nil {
		destTensor = tensorFactory(destType, shape)
	} else {
		destTensor = tensor.New(destType, shape)
	}

	// Copy with type conversion (element by element)
	size := srcTensor.Size()
	for i := 0; i < size; i++ {
		val := srcTensor.At(i)
		destTensor.SetAt(val, i)
	}

	return destTensor, nil
}

func gobToParameter(gp gobParameter, tensorFactory func(types.DataType, types.Shape) types.Tensor) types.Parameter {
	result := types.Parameter{
		RequiresGrad: gp.RequiresGrad,
	}

	if len(gp.Data.Shape) > 0 {
		result.Data = gobToTensor(gp.Data, tensorFactory)
	}
	if len(gp.Grad.Shape) > 0 {
		result.Grad = gobToTensor(gp.Grad, tensorFactory)
	}

	return result
}
