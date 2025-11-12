package json

import (
	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

// Internal structs for JSON encoding/decoding.
// These structs provide proper JSON representation for domain types.

// jsonTensor represents a tensor for JSON encoding/decoding.
type jsonTensor struct {
	DataType string  `json:"dtype"`
	Shape    []int   `json:"shape"`
	Data     any     `json:"data"` // JSON will handle type-specific encoding
}

// jsonParameter represents a parameter for JSON encoding/decoding.
type jsonParameter struct {
	Data         *jsonTensor `json:"data,omitempty"`
	Grad         *jsonTensor `json:"grad,omitempty"`
	RequiresGrad bool        `json:"requires_grad"`
}

// jsonLayer represents a layer for JSON encoding/decoding.
type jsonLayer struct {
	Name       string                   `json:"name"`
	Type       string                   `json:"type"`
	CanLearn   bool                     `json:"can_learn"`
	InputShape []int                    `json:"input_shape,omitempty"`
	Parameters map[string]jsonParameter `json:"parameters,omitempty"`
}

// jsonModel represents a model for JSON encoding/decoding.
type jsonModel struct {
	Name       string                   `json:"name"`
	Type       string                   `json:"type"`
	CanLearn   bool                     `json:"can_learn"`
	LayerCount int                      `json:"layer_count"`
	Layers     []jsonLayer              `json:"layers,omitempty"`
	Parameters map[string]jsonParameter `json:"parameters,omitempty"`
}

// jsonValue is a discriminated union for different value types.
type jsonValue struct {
	Kind string `json:"kind"` // "tensor", "layer", "model", "slice", "generic"

	// Type-specific fields
	Tensor *jsonTensor `json:"tensor,omitempty"`
	Layer  *jsonLayer  `json:"layer,omitempty"`
	Model  *jsonModel  `json:"model,omitempty"`

	// For slices/arrays
	SliceType string `json:"slice_type,omitempty"`
	SliceData any    `json:"slice_data,omitempty"`
}

// Helper functions to convert between domain types and JSON structs

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

func tensorToJSON(t types.Tensor) *jsonTensor {
	if t.Empty() {
		return nil
	}

	return &jsonTensor{
		DataType: dtypeToString(t.DataType()),
		Shape:    t.Shape(),
		Data:     t.Data(),
	}
}

func parameterToJSON(p types.Parameter) jsonParameter {
	result := jsonParameter{
		RequiresGrad: p.RequiresGrad,
	}

	if !p.Data.Empty() {
		result.Data = tensorToJSON(p.Data)
	}
	if !p.Grad.Empty() {
		result.Grad = tensorToJSON(p.Grad)
	}

	return result
}

func layerToJSON(layer types.Layer) jsonLayer {
	result := jsonLayer{
		Name:       layer.Name(),
		CanLearn:   layer.CanLearn(),
		Parameters: make(map[string]jsonParameter),
	}

	// Get input shape if available
	input := layer.Input()
	if !input.Empty() {
		result.InputShape = input.Shape()
	}

	// Convert parameters
	params := layer.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToJSON(param)
	}

	return result
}

func modelToJSON(model types.Model) jsonModel {
	result := jsonModel{
		Name:       model.Name(),
		CanLearn:   model.CanLearn(),
		LayerCount: model.LayerCount(),
		Layers:     make([]jsonLayer, 0),
		Parameters: make(map[string]jsonParameter),
	}

	// Convert layers
	for i := 0; i < model.LayerCount(); i++ {
		layer := model.GetLayer(i)
		if layer != nil {
			result.Layers = append(result.Layers, layerToJSON(layer))
		}
	}

	// Convert parameters
	params := model.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToJSON(param)
	}

	return result
}

