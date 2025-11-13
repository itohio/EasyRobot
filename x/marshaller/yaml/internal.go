package yaml

import (
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Internal structs for YAML encoding/decoding.
// YAML uses same structure as JSON since YAML is a superset of JSON.

// yamlTensor represents a tensor for YAML encoding/decoding.
type yamlTensor struct {
	DataType string `yaml:"dtype"`
	Shape    []int  `yaml:"shape"`
	Data     any    `yaml:"data"`
}

// yamlParameter represents a parameter for YAML encoding/decoding.
type yamlParameter struct {
	Data         *yamlTensor `yaml:"data,omitempty"`
	Grad         *yamlTensor `yaml:"grad,omitempty"`
	RequiresGrad bool        `yaml:"requires_grad"`
}

// yamlLayer represents a layer for YAML encoding/decoding.
type yamlLayer struct {
	Name       string                   `yaml:"name"`
	Type       string                   `yaml:"type"`
	CanLearn   bool                     `yaml:"can_learn"`
	InputShape []int                    `yaml:"input_shape,omitempty"`
	Parameters map[string]yamlParameter `yaml:"parameters,omitempty"`
}

// yamlModel represents a model for YAML encoding/decoding.
type yamlModel struct {
	Name       string                   `yaml:"name"`
	Type       string                   `yaml:"type"`
	CanLearn   bool                     `yaml:"can_learn"`
	LayerCount int                      `yaml:"layer_count"`
	Layers     []yamlLayer              `yaml:"layers,omitempty"`
	Parameters map[string]yamlParameter `yaml:"parameters,omitempty"`
}

// yamlValue is a discriminated union for different value types.
type yamlValue struct {
	Kind string `yaml:"kind"`

	// Type-specific fields
	Tensor *yamlTensor `yaml:"tensor,omitempty"`
	Layer  *yamlLayer  `yaml:"layer,omitempty"`
	Model  *yamlModel  `yaml:"model,omitempty"`

	// For slices/arrays
	SliceType string `yaml:"slice_type,omitempty"`
	SliceData any    `yaml:"slice_data,omitempty"`
}

// Helper functions - reuse from common logic

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

func tensorToYAML(t types.Tensor) *yamlTensor {
	if t.Empty() {
		return nil
	}

	return &yamlTensor{
		DataType: dtypeToString(t.DataType()),
		Shape:    t.Shape(),
		Data:     t.Data(),
	}
}

func parameterToYAML(p types.Parameter) yamlParameter {
	result := yamlParameter{
		RequiresGrad: p.RequiresGrad,
	}

	if !p.Data.Empty() {
		result.Data = tensorToYAML(p.Data)
	}
	if !p.Grad.Empty() {
		result.Grad = tensorToYAML(p.Grad)
	}

	return result
}

func layerToYAML(layer types.Layer) yamlLayer {
	result := yamlLayer{
		Name:       layer.Name(),
		CanLearn:   layer.CanLearn(),
		Parameters: make(map[string]yamlParameter),
	}

	// Get input shape if available
	input := layer.Input()
	if !input.Empty() {
		result.InputShape = input.Shape()
	}

	// Convert parameters
	params := layer.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToYAML(param)
	}

	return result
}

func modelToYAML(model types.Model) yamlModel {
	result := yamlModel{
		Name:       model.Name(),
		CanLearn:   model.CanLearn(),
		LayerCount: model.LayerCount(),
		Layers:     make([]yamlLayer, 0),
		Parameters: make(map[string]yamlParameter),
	}

	// Convert layers
	for i := 0; i < model.LayerCount(); i++ {
		layer := model.GetLayer(i)
		if layer != nil {
			result.Layers = append(result.Layers, layerToYAML(layer))
		}
	}

	// Convert parameters
	params := model.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToYAML(param)
	}

	return result
}
