package marshallerv1

import (
	"fmt"
	"io"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	pb "github.com/itohio/EasyRobot/types/core"
	"google.golang.org/protobuf/proto"
)

// Marshaller implements protobuf-based marshalling.
type Marshaller struct {
	opts types.Options
}

// NewMarshaller creates a new protobuf marshaller.
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
	return "protobuf"
}

// Marshal encodes a value to protobuf format.
func (m *Marshaller) Marshal(dst io.Writer, value any, opts ...types.Option) error {
	// Apply additional options
	for _, opt := range opts {
		opt.Apply(&m.opts)
	}
	// If value is already a proto.Message, marshal it directly
	if msg, ok := value.(proto.Message); ok {
		data, err := proto.Marshal(msg)
		if err != nil {
			return fmt.Errorf("protobuf marshal: %w", err)
		}
		_, err = dst.Write(data)
		return err
	}

	// Otherwise, convert to our Value wrapper
	protoValue, err := m.valueToProto(value)
	if err != nil {
		return fmt.Errorf("protobuf convert: %w", err)
	}

	data, err := proto.Marshal(protoValue)
	if err != nil {
		return fmt.Errorf("protobuf marshal: %w", err)
	}

	_, err = dst.Write(data)
	return err
}

func (m *Marshaller) valueToProto(value any) (*pb.Value, error) {
	if value == nil {
		return nil, fmt.Errorf("cannot marshal nil value")
	}

	result := &pb.Value{}

	// Check for Model first (since Model is also Layer)
	if model, ok := value.(types.Model); ok {
		result.Kind = "model"
		modelProto, err := modelToProto(model)
		if err != nil {
			return nil, err
		}
		result.Data = &pb.Value_Model{Model: modelProto}
		return result, nil
	}

	// Check for Layer
	if layer, ok := value.(types.Layer); ok {
		result.Kind = "layer"
		layerProto, err := layerToProto(layer)
		if err != nil {
			return nil, err
		}
		result.Data = &pb.Value_Layer{Layer: layerProto}
		return result, nil
	}

	// Check for Tensor
	if tensor, ok := value.(types.Tensor); ok {
		result.Kind = "tensor"
		tensorProto, err := tensorToProto(tensor)
		if err != nil {
			return nil, err
		}
		result.Data = &pb.Value_Tensor{Tensor: tensorProto}
		return result, nil
	}

	// Note: Matrix and Vector from gonum cannot be marshalled generically here
	// as they don't provide data access methods. Users should either:
	// 1. Marshal them as proto.Message if they have proto representations
	// 2. Convert them to slices first
	// 3. Use a format-specific custom marshaller

	// Check for slices/arrays of numeric types
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		result.Kind = "slice"
		sliceProto, err := m.sliceToProto(value)
		if err != nil {
			return nil, err
		}
		result.Data = &pb.Value_Slice{Slice: sliceProto}
		return result, nil
	}

	return nil, fmt.Errorf("unsupported type: %T", value)
}

func (m *Marshaller) sliceToProto(value any) (*pb.Slice, error) {
	rv := reflect.ValueOf(value)
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		return nil, fmt.Errorf("not a slice or array: %T", value)
	}

	result := &pb.Slice{
		Type: rv.Type().Elem().String(),
	}

	// Convert based on element type
	switch value.(type) {
	case []float32:
		data := value.([]float32)
		result.DataF32 = data
	case []float64:
		data := value.([]float64)
		result.DataF64 = data
	case []int:
		data := value.([]int)
		result.DataI64 = make([]int64, len(data))
		for i, v := range data {
			result.DataI64[i] = int64(v)
		}
	case []int32:
		data := value.([]int32)
		result.DataI32 = data
	case []int64:
		data := value.([]int64)
		result.DataI64 = data
	case []byte:
		data := value.([]byte)
		result.DataBytes = data
	default:
		return nil, fmt.Errorf("unsupported slice type: %T", value)
	}

	return result, nil
}
