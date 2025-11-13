package marshallerv1

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	pb "github.com/itohio/EasyRobot/types/core"
)

// Conversion helpers between domain types and protobuf messages

func dtypeToString(dt types.DataType) string {
	switch dt {
	case types.FP32:
		return "fp32"
	case types.FP64:
		return "fp64"
	case types.INT8:
		return "int8"
	case types.INT16:
		return "int16"
	case types.INT32:
		return "int32"
	case types.INT64:
		return "int64"
	case types.INT:
		return "int"
	case types.FP16:
		return "fp16"
	case types.INT48:
		return "int48"
	default:
		return "unknown"
	}
}

func stringToDtype(s string) types.DataType {
	switch s {
	case "fp32":
		return types.FP32
	case "fp64":
		return types.FP64
	case "int8":
		return types.INT8
	case "int16":
		return types.INT16
	case "int32":
		return types.INT32
	case "int64":
		return types.INT64
	case "int":
		return types.INT
	case "fp16":
		return types.FP16
	case "int48":
		return types.INT48
	default:
		return types.DT_UNKNOWN
	}
}

func tensorToProto(t types.Tensor) (*pb.Tensor, error) {
	if t.Empty() {
		return nil, nil
	}

	// Convert shape
	shape := make([]int32, len(t.Shape()))
	for i, s := range t.Shape() {
		shape[i] = int32(s)
	}

	result := &pb.Tensor{
		Dtype: dtypeToString(t.DataType()),
		Shape: shape,
	}

	// Store data in appropriate field based on type
	dtype := t.DataType()
	data := t.Data()

	switch dtype {
	case types.FP32:
		if fp32Data, ok := data.([]float32); ok {
			result.DataF32 = fp32Data
		}
	case types.FP64:
		if fp64Data, ok := data.([]float64); ok {
			result.DataF64 = fp64Data
		}
	case types.INT32:
		if int32Data, ok := data.([]int32); ok {
			result.DataI32 = int32Data
		}
	case types.INT64:
		if int64Data, ok := data.([]int64); ok {
			result.DataI64 = int64Data
		}
	case types.INT:
		if intData, ok := data.([]int); ok {
			// Convert int to int64 for portability
			result.DataI64 = make([]int64, len(intData))
			for i, v := range intData {
				result.DataI64[i] = int64(v)
			}
		}
	}

	return result, nil
}

func protoToTensor(pt *pb.Tensor, destType types.DataType, tensorFactory func(types.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if pt == nil || len(pt.Shape) == 0 {
		return nil, fmt.Errorf("invalid tensor proto")
	}

	// Convert shape
	shape := make(types.Shape, len(pt.Shape))
	for i, s := range pt.Shape {
		shape[i] = int(s)
	}

	dtype := stringToDtype(pt.Dtype)

	// Create tensor
	var t types.Tensor
	if destType != 0 {
		t = tensorFactory(destType, shape)
	} else {
		t = tensorFactory(dtype, shape)
	}

	// Copy data from appropriate field based on source dtype
	switch dtype {
	case types.FP32:
		if len(pt.DataF32) > 0 {
			// If types match, copy directly
			if destType == 0 || destType == dtype {
				if destData, ok := t.Data().([]float32); ok {
					copy(destData, pt.DataF32)
				}
			} else {
				// Type conversion needed - use SetAt for each element
				for i, v := range pt.DataF32 {
					t.SetAt(float64(v), i)
				}
			}
		}
	case types.FP64:
		if len(pt.DataF64) > 0 {
			if destType == 0 || destType == dtype {
				if destData, ok := t.Data().([]float64); ok {
					copy(destData, pt.DataF64)
				}
			} else {
				for i, v := range pt.DataF64 {
					t.SetAt(v, i)
				}
			}
		}
	case types.INT32:
		if len(pt.DataI32) > 0 {
			if destType == 0 || destType == dtype {
				if destData, ok := t.Data().([]int32); ok {
					copy(destData, pt.DataI32)
				}
			} else {
				for i, v := range pt.DataI32 {
					t.SetAt(float64(v), i)
				}
			}
		}
	case types.INT64:
		if len(pt.DataI64) > 0 {
			if destType == 0 || destType == dtype {
				if destData, ok := t.Data().([]int64); ok {
					copy(destData, pt.DataI64)
				}
			} else {
				for i, v := range pt.DataI64 {
					t.SetAt(float64(v), i)
				}
			}
		}
	case types.INT:
		if len(pt.DataI64) > 0 {
			if destType == 0 || destType == dtype {
				if destData, ok := t.Data().([]int); ok {
					for i, v := range pt.DataI64 {
						destData[i] = int(v)
					}
				}
			} else {
				for i, v := range pt.DataI64 {
					t.SetAt(float64(v), i)
				}
			}
		}
	}

	return t, nil
}

func parameterToProto(p types.Parameter) (*pb.Parameter, error) {
	result := &pb.Parameter{
		RequiresGrad: p.RequiresGrad,
	}

	if !p.Data.Empty() {
		data, err := tensorToProto(p.Data)
		if err != nil {
			return nil, err
		}
		result.Data = data
	}
	if !p.Grad.Empty() {
		grad, err := tensorToProto(p.Grad)
		if err != nil {
			return nil, err
		}
		result.Grad = grad
	}

	return result, nil
}

func layerToProto(layer types.Layer) (*pb.Layer, error) {
	result := &pb.Layer{
		Name:       layer.Name(),
		CanLearn:   layer.CanLearn(),
		Parameters: make(map[int32]*pb.Parameter),
	}

	// Get input shape if available
	input := layer.Input()
	if !input.Empty() {
		shape := input.Shape()
		result.InputShape = make([]int32, len(shape))
		for i, s := range shape {
			result.InputShape[i] = int32(s)
		}
	}

	// Convert parameters
	params := layer.Parameters()
	for idx, param := range params {
		p, err := parameterToProto(param)
		if err != nil {
			return nil, err
		}
		result.Parameters[int32(idx)] = p
	}

	return result, nil
}

func modelToProto(model types.Model) (*pb.Model, error) {
	result := &pb.Model{
		Name:       model.Name(),
		CanLearn:   model.CanLearn(),
		Layers:     make([]*pb.Layer, 0),
		Parameters: make(map[int32]*pb.Parameter),
	}

	// Convert layers
	for i := 0; i < model.LayerCount(); i++ {
		layer := model.GetLayer(i)
		if layer != nil {
			l, err := layerToProto(layer)
			if err != nil {
				return nil, err
			}
			result.Layers = append(result.Layers, l)
		}
	}

	// Convert parameters
	params := model.Parameters()
	for idx, param := range params {
		p, err := parameterToProto(param)
		if err != nil {
			return nil, err
		}
		result.Parameters[int32(idx)] = p
	}

	return result, nil
}
