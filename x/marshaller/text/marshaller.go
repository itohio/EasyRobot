package text

import (
	"fmt"
	"io"
	"reflect"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

// Marshaller implements text-based marshalling that prints object information.
// It does not have an unmarshaller counterpart.
type Marshaller struct {
	opts types.Options
}

// New creates a new text marshaller.
func New(opts ...types.Option) types.Marshaller {
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
	return "text"
}

// Marshal writes a text representation of the value to w.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	// Apply additional options
	localOpts := m.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	if value == nil {
		_, err := fmt.Fprintf(w, "nil\n")
		return err
	}

	// Check for Model first (Model is also Layer)
	if model, ok := value.(types.Model); ok {
		return m.marshalModel(w, model)
	}

	// Then check for Layer
	if layer, ok := value.(types.Layer); ok {
		return m.marshalLayer(w, layer)
	}

	// Check for Tensor
	if tensor, ok := value.(types.Tensor); ok {
		return m.marshalTensor(w, tensor)
	}

	// Check for arrays/slices
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		return m.marshalSlice(w, rv)
	}

	// Fallback to generic representation
	return m.marshalGeneric(w, value)
}

func (m *Marshaller) marshalTensor(w io.Writer, t types.Tensor) error {
	shape := t.Shape()
	dtype := t.DataType()
	size := t.Size()

	_, err := fmt.Fprintf(w, "Tensor(shape=%v, dtype=%s, size=%d)\n", shape, dtypeString(dtype), size)
	return err
}

func (m *Marshaller) marshalLayer(w io.Writer, layer types.Layer) error {
	name := layer.Name()
	layerType := reflect.TypeOf(layer).String()

	// Get input/output shapes
	input := layer.Input()
	output := layer.Output()

	var inputShape, outputShape types.Shape
	if !input.Empty() {
		inputShape = input.Shape()
	}
	if !output.Empty() {
		outputShape = output.Shape()
	}

	// Count parameters
	params := layer.Parameters()
	totalParams := 0
	trainableParams := 0
	for _, param := range params {
		pSize := param.Data.Size()
		totalParams += pSize
		if param.RequiresGrad {
			trainableParams += pSize
		}
	}

	_, err := fmt.Fprintf(w, "Layer: %s\n", name)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "  Type: %s\n", layerType)
	if err != nil {
		return err
	}
	if inputShape != nil {
		_, err = fmt.Fprintf(w, "  Input shape: %v\n", inputShape)
		if err != nil {
			return err
		}
	}
	if outputShape != nil {
		_, err = fmt.Fprintf(w, "  Output shape: %v\n", outputShape)
		if err != nil {
			return err
		}
	}
	_, err = fmt.Fprintf(w, "  Parameters: %d (trainable: %d)\n", totalParams, trainableParams)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "  Can learn: %v\n", layer.CanLearn())
	return err
}

func (m *Marshaller) marshalModel(w io.Writer, model types.Model) error {
	name := model.Name()
	modelType := reflect.TypeOf(model).String()
	layerCount := model.LayerCount()

	// Count total parameters
	params := model.Parameters()
	totalParams := 0
	trainableParams := 0
	for _, param := range params {
		pSize := param.Data.Size()
		totalParams += pSize
		if param.RequiresGrad {
			trainableParams += pSize
		}
	}

	_, err := fmt.Fprintf(w, "Model: %s\n", name)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "  Type: %s\n", modelType)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "  Layers: %d\n", layerCount)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "  Total parameters: %d (trainable: %d)\n", totalParams, trainableParams)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "  Can learn: %v\n", model.CanLearn())
	if err != nil {
		return err
	}

	// Print each layer
	_, err = fmt.Fprintf(w, "\nLayers:\n")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "%-40s %-30s %-20s %-20s %s\n", "Name", "Type", "Input Shape", "Output Shape", "Parameters")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "%s\n", repeatChar('=', 150))
	if err != nil {
		return err
	}

	for i := 0; i < layerCount; i++ {
		layer := model.GetLayer(i)
		if layer == nil {
			continue
		}

		layerName := layer.Name()
		layerType := reflect.TypeOf(layer).String()

		// Get shapes
		input := layer.Input()
		output := layer.Output()

		var inputShape, outputShape string
		if !input.Empty() {
			inputShape = fmt.Sprintf("%v", input.Shape())
		} else {
			inputShape = "N/A"
		}
		if !output.Empty() {
			outputShape = fmt.Sprintf("%v", output.Shape())
		} else {
			outputShape = "N/A"
		}

		// Count layer parameters
		layerParams := layer.Parameters()
		layerParamCount := 0
		for _, param := range layerParams {
			layerParamCount += param.Data.Size()
		}

		_, err = fmt.Fprintf(w, "%-40s %-30s %-20s %-20s %d\n",
			truncate(layerName, 40),
			truncate(layerType, 30),
			truncate(inputShape, 20),
			truncate(outputShape, 20),
			layerParamCount)
		if err != nil {
			return err
		}
	}

	_, err = fmt.Fprintf(w, "%s\n", repeatChar('=', 150))
	return err
}

func (m *Marshaller) marshalSlice(w io.Writer, rv reflect.Value) error {
	elemType := rv.Type().Elem()
	length := rv.Len()

	_, err := fmt.Fprintf(w, "Array/Slice(type=%s, length=%d)\n", elemType.String(), length)
	return err
}

func (m *Marshaller) marshalGeneric(w io.Writer, value any) error {
	rv := reflect.ValueOf(value)
	_, err := fmt.Fprintf(w, "Value(type=%s, kind=%s)\n", rv.Type().String(), rv.Kind().String())
	return err
}

func dtypeString(dt types.DataType) string {
	switch dt {
	case types.FP32:
		return "FP32"
	case types.FP64:
		return "FP64"
	case types.INT8:
		return "INT8"
	case types.INT16:
		return "INT16"
	case types.INT32:
		return "INT32"
	case types.INT64:
		return "INT64"
	case types.INT:
		return "INT"
	case types.FP16:
		return "FP16"
	case types.INT48:
		return "INT48"
	default:
		return "UNKNOWN"
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func repeatChar(ch rune, count int) string {
	result := make([]rune, count)
	for i := range result {
		result[i] = ch
	}
	return string(result)
}
