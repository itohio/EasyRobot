package yaml

import (
	"io"
	"reflect"

	"gopkg.in/yaml.v3"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Marshaller implements YAML-based marshalling.
type Marshaller struct {
	opts types.Options
}

// NewMarshaller creates a new YAML marshaller.
func NewMarshaller(opts ...types.Option) types.Marshaller {
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
	return "yaml"
}

// Marshal encodes value to YAML format.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	// Apply additional options
	localOpts := m.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	if value == nil {
		return types.NewError("marshal", "yaml", "nil value", nil)
	}

	encoder := yaml.NewEncoder(w)
	encoder.SetIndent(2)
	defer encoder.Close()

	// Convert value to yamlValue
	yv, err := m.valueToYAML(value)
	if err != nil {
		return types.NewError("marshal", "yaml", "value conversion", err)
	}

	// Encode yamlValue
	if err := encoder.Encode(yv); err != nil {
		return types.NewError("marshal", "yaml", "encoding", err)
	}

	return nil
}

func (m *Marshaller) valueToYAML(value any) (*yamlValue, error) {
	if value == nil {
		return nil, types.NewError("marshal", "yaml", "nil value", nil)
	}

	// Check for Model first (Model is also Layer)
	if model, ok := value.(types.Model); ok {
		ym := modelToYAML(model)
		ym.Type = reflect.TypeOf(value).String()
		return &yamlValue{
			Kind:  "model",
			Model: &ym,
		}, nil
	}

	// Then check for Layer
	if layer, ok := value.(types.Layer); ok {
		yl := layerToYAML(layer)
		yl.Type = reflect.TypeOf(value).String()
		return &yamlValue{
			Kind:  "layer",
			Layer: &yl,
		}, nil
	}

	// Check for Tensor
	if tensor, ok := value.(types.Tensor); ok {
		yt := tensorToYAML(tensor)
		return &yamlValue{
			Kind:   "tensor",
			Tensor: yt,
		}, nil
	}

	// Check for slices/arrays
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		return &yamlValue{
			Kind:      "slice",
			SliceType: rv.Type().String(),
			SliceData: value,
		}, nil
	}

	// Fallback to generic encoding
	return &yamlValue{
		Kind: "generic",
	}, nil
}
