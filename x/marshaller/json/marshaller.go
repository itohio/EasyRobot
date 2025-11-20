package json

import (
	"encoding/json"
	"io"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
)

// Marshaller implements JSON-based marshalling.
type Marshaller struct {
	opts types.Options
}

// NewMarshaller creates a new JSON marshaller.
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
	return "json"
}

// Marshal encodes value to JSON format.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	// Apply additional options
	localOpts := m.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	if value == nil {
		return types.NewError("marshal", "json", "nil value", nil)
	}

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ") // Pretty print

	// Convert value to jsonValue
	jv, err := m.valueToJSON(value)
	if err != nil {
		return types.NewError("marshal", "json", "value conversion", err)
	}

	// Encode jsonValue
	if err := encoder.Encode(jv); err != nil {
		return types.NewError("marshal", "json", "encoding", err)
	}

	return nil
}

func (m *Marshaller) valueToJSON(value any) (*jsonValue, error) {
	if value == nil {
		return nil, types.NewError("marshal", "json", "nil value", nil)
	}

	// Check for Model first (Model is also Layer)
	if model, ok := value.(types.Model); ok {
		jm := modelToJSON(model)
		jm.Type = reflect.TypeOf(value).String()
		return &jsonValue{
			Kind:  "model",
			Model: &jm,
		}, nil
	}

	// Then check for Layer
	if layer, ok := value.(types.Layer); ok {
		jl := layerToJSON(layer)
		jl.Type = reflect.TypeOf(value).String()
		return &jsonValue{
			Kind:  "layer",
			Layer: &jl,
		}, nil
	}

	// Check for Tensor
	if tensor, ok := value.(types.Tensor); ok {
		jt := tensorToJSON(tensor)
		return &jsonValue{
			Kind:   "tensor",
			Tensor: jt,
		}, nil
	}

	// Check for Graph
	if graph, ok := value.(graph.Graph[any, any]); ok {
		jg, err := graphToJSON(graph)
		if err != nil {
			return nil, types.NewError("marshal", "json", "graph conversion", err)
		}
		kind := "graph"
		if jg.Metadata != nil {
			kind = jg.Metadata.Kind
		}
		return &jsonValue{
			Kind:  kind,
			Graph: jg,
		}, nil
	}

	// Check for slices/arrays
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		return &jsonValue{
			Kind:      "slice",
			SliceType: rv.Type().String(),
			SliceData: value,
		}, nil
	}

	// Check for graphs using reflection
	if reflect.TypeOf(value).String() != "" { // Always true, just to use reflection
		if jg, err := graphToJSON(value); err == nil && jg != nil {
			kind := "graph"
			if jg.Metadata != nil {
				kind = jg.Metadata.Kind
			}
			return &jsonValue{
				Kind:  kind,
				Graph: jg,
			}, nil
		}
	}

	// Fallback to generic encoding
	return &jsonValue{
		Kind: "generic",
	}, nil
}
