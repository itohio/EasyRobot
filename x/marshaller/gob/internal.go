package gob

import (
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Internal structs for gob encoding/decoding.
// These structs encapsulate domain types to avoid directly serializing
// into mat/vec/tensor/nn structs, always using interfaces instead.

// gobTensor represents a tensor for gob encoding/decoding.
type gobTensor struct {
	DataType uint8 // tensor.DataType as uint8
	Shape    []int // tensor.Shape
	// Data stored as interface{} - gob will handle the type-specific encoding
	Data any
}

// gobParameter represents a parameter for gob encoding/decoding.
type gobParameter struct {
	Data         gobTensor
	Grad         gobTensor
	RequiresGrad bool
}

// gobLayer represents a layer for gob encoding/decoding.
type gobLayer struct {
	Name       string
	Type       string // full type name from reflection
	CanLearn   bool
	InputShape []int
	Parameters map[int]gobParameter // ParamIndex -> Parameter
	// Layer-specific data stored as raw bytes
	ExtraData []byte
}

// gobModel represents a model for gob encoding/decoding.
type gobModel struct {
	Name       string
	Type       string // full type name from reflection
	CanLearn   bool
	LayerCount int
	Layers     []gobLayer
	Parameters map[int]gobParameter // ParamIndex -> Parameter
	// Model-specific data stored as raw bytes
	ExtraData []byte
}

// gobValue is a discriminated union for different value types.
type gobValue struct {
	Kind string // "tensor", "layer", "model", "slice", "generic"

	// Type-specific fields
	Tensor *gobTensor
	Layer  *gobLayer
	Model  *gobModel

	// For slices/arrays
	SliceType string
	SliceData any
}

// Helper functions to convert between domain types and gob structs

func tensorToGob(t types.Tensor) gobTensor {
	if t.Empty() {
		return gobTensor{}
	}

	return gobTensor{
		DataType: uint8(t.DataType()),
		Shape:    t.Shape(),
		Data:     t.Data(), // gob will handle the encoding
	}
}

func parameterToGob(p types.Parameter) gobParameter {
	result := gobParameter{
		RequiresGrad: p.RequiresGrad,
	}

	if !p.Data.Empty() {
		result.Data = tensorToGob(p.Data)
	}
	if !p.Grad.Empty() {
		result.Grad = tensorToGob(p.Grad)
	}

	return result
}

func layerToGob(layer types.Layer) gobLayer {
	result := gobLayer{
		Name:       layer.Name(),
		CanLearn:   layer.CanLearn(),
		Parameters: make(map[int]gobParameter),
	}

	// Get input shape if available
	input := layer.Input()
	if !input.Empty() {
		result.InputShape = input.Shape()
	}

	// Convert parameters
	params := layer.Parameters()
	for idx, param := range params {
		result.Parameters[int(idx)] = parameterToGob(param)
	}

	return result
}

func modelToGob(model types.Model) gobModel {
	result := gobModel{
		Name:       model.Name(),
		CanLearn:   model.CanLearn(),
		LayerCount: model.LayerCount(),
		Layers:     make([]gobLayer, 0),
		Parameters: make(map[int]gobParameter),
	}

	// Convert layers
	for i := 0; i < model.LayerCount(); i++ {
		layer := model.GetLayer(i)
		if layer != nil {
			result.Layers = append(result.Layers, layerToGob(layer))
		}
	}

	// Convert parameters
	params := model.Parameters()
	for idx, param := range params {
		result.Parameters[int(idx)] = parameterToGob(param)
	}

	return result
}
