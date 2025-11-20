package gob

import (
	"encoding/gob"
	"io"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Marshaller implements gob-based marshalling.
type Marshaller struct {
	opts types.Options
}

// NewMarshaller creates a new gob marshaller.
func NewMarshaller(opts ...types.Option) *Marshaller {
	m := &Marshaller{
		opts: types.Options{},
	}
	for _, opt := range opts {
		opt.Apply(&m.opts)
	}
	return m
}

// Format returns the format name.
func (m *Marshaller) Format() string {
	return "gob"
}

// Marshal encodes value to gob format.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	// Apply additional options
	localOpts := m.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	if value == nil {
		return types.NewError("marshal", "gob", "nil value", nil)
	}

	encoder := gob.NewEncoder(w)

	// Convert value to gobValue
	gv, err := m.valueToGob(value)
	if err != nil {
		return types.NewError("marshal", "gob", "value conversion", err)
	}

	// Encode gobValue
	if err := encoder.Encode(gv); err != nil {
		return types.NewError("marshal", "gob", "encoding", err)
	}

	return nil
}

func (m *Marshaller) valueToGob(value any) (*gobValue, error) {
	if value == nil {
		return nil, types.NewError("marshal", "gob", "nil value", nil)
	}

	// Check for Model first (Model is also Layer)
	if model, ok := value.(types.Model); ok {
		gm := modelToGob(model)
		gm.Type = reflect.TypeOf(value).String()
		return &gobValue{
			Kind:  "model",
			Model: &gm,
		}, nil
	}

	// Then check for Layer
	if layer, ok := value.(types.Layer); ok {
		gl := layerToGob(layer)
		gl.Type = reflect.TypeOf(value).String()
		return &gobValue{
			Kind:  "layer",
			Layer: &gl,
		}, nil
	}

	// Check for Tensor
	if tensor, ok := value.(types.Tensor); ok {
		gt := tensorToGob(tensor)
		return &gobValue{
			Kind:   "tensor",
			Tensor: &gt,
		}, nil
	}

	// Check for slices/arrays
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		return m.sliceToGob(rv)
	}

	// Fallback to generic gob encoding
	// For generic values, we just use gob's default encoding
	return &gobValue{
		Kind: "generic",
		// For generic values, we don't convert them - gob will handle it
	}, nil
}

func (m *Marshaller) sliceToGob(rv reflect.Value) (*gobValue, error) {
	// For slices, just store the data directly - gob will handle encoding
	return &gobValue{
		Kind:      "slice",
		SliceType: rv.Type().String(),
		SliceData: rv.Interface(),
	}, nil
}
