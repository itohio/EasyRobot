package marshallerv1

import (
	"fmt"
	"io"
	"reflect"

	pb "github.com/itohio/EasyRobot/types/core"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"google.golang.org/protobuf/proto"
)

// Unmarshaller implements protobuf-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new protobuf unmarshaller.
func NewUnmarshaller(opts ...types.Option) *Unmarshaller {
	u := &Unmarshaller{
		opts: types.Options{},
	}
	for _, opt := range opts {
		opt.Apply(&u.opts)
	}
	return u
}

// Format returns the format name.
func (u *Unmarshaller) Format() string {
	return "protobuf"
}

// Unmarshal decodes a value from protobuf format.
func (u *Unmarshaller) Unmarshal(src io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	for _, opt := range opts {
		opt.Apply(&u.opts)
	}
	// Read all data
	data, err := io.ReadAll(src)
	if err != nil {
		return fmt.Errorf("protobuf read: %w", err)
	}

	// If dst is a proto.Message, unmarshal directly
	if msg, ok := dst.(proto.Message); ok {
		return proto.Unmarshal(data, msg)
	}

	// Otherwise, unmarshal to our Value wrapper and convert
	var protoValue pb.Value
	if err := proto.Unmarshal(data, &protoValue); err != nil {
		return fmt.Errorf("protobuf unmarshal: %w", err)
	}

	value, err := u.protoToValue(&protoValue)
	if err != nil {
		return fmt.Errorf("protobuf convert: %w", err)
	}

	// Assign to destination
	return u.assignToDst(dst, value)
}

func (u *Unmarshaller) protoToValue(pv *pb.Value) (any, error) {
	tensorFactory := u.opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dt types.DataType, sh types.Shape) types.Tensor {
			return tensor.New(tensortypes.DataType(dt), shape.NewShape(sh...))
		}
	}

	switch pv.Kind {
	case "tensor":
		tensorProto := pv.GetTensor()
		if tensorProto == nil {
			return nil, fmt.Errorf("tensor value is nil")
		}
		return protoToTensor(tensorProto, u.opts.DestinationType, tensorFactory)

	case "matrix":
		// Note: We can't reconstruct mat.Matrix without knowing the implementation
		// Return the protobuf message for now
		return pv.GetMatrix(), nil

	case "vector":
		// Note: We can't reconstruct vec.Vector without knowing the implementation
		// Return the protobuf message for now
		return pv.GetVector(), nil

	case "layer":
		// Note: We can't reconstruct nn.Layer without type information
		// Return the protobuf message for now
		return pv.GetLayer(), nil

	case "model":
		// Note: We can't reconstruct nn.Model without type information
		// Return the protobuf message for now
		return pv.GetModel(), nil

	case "slice":
		sliceProto := pv.GetSlice()
		if sliceProto == nil {
			return nil, fmt.Errorf("slice value is nil")
		}
		return u.protoToSlice(sliceProto)

	default:
		return nil, fmt.Errorf("unknown kind: %s", pv.Kind)
	}
}

func (u *Unmarshaller) protoToSlice(ps *pb.Slice) (any, error) {
	switch ps.Type {
	case "float32":
		return ps.DataF32, nil
	case "float64":
		return ps.DataF64, nil
	case "complex64":
		// Complex64 is stored as pairs of float32 [real0, imag0, real1, imag1, ...]
		if len(ps.DataF32)%2 != 0 {
			return nil, fmt.Errorf("invalid complex64 data: odd number of float32 values")
		}
		result := make([]complex64, len(ps.DataF32)/2)
		for i := 0; i < len(result); i++ {
			result[i] = complex(ps.DataF32[i*2], ps.DataF32[i*2+1])
		}
		return result, nil
	case "int32":
		return ps.DataI32, nil
	case "int64":
		return ps.DataI64, nil
	case "int":
		// Convert int64 to int
		result := make([]int, len(ps.DataI64))
		for i, v := range ps.DataI64 {
			result[i] = int(v)
		}
		return result, nil
	case "uint8":
		return ps.DataBytes, nil
	default:
		return nil, fmt.Errorf("unsupported slice type: %s", ps.Type)
	}
}

func (u *Unmarshaller) assignToDst(dst any, value any) error {
	dstVal := reflect.ValueOf(dst)
	if dstVal.Kind() != reflect.Ptr {
		return fmt.Errorf("destination must be a pointer")
	}

	dstElem := dstVal.Elem()
	srcVal := reflect.ValueOf(value)

	// If dst is an interface, we need to set it to the concrete value
	if dstElem.Kind() == reflect.Interface {
		if !srcVal.Type().AssignableTo(dstElem.Type()) {
			return fmt.Errorf("cannot assign %T to %T", value, dst)
		}
		dstElem.Set(srcVal)
		return nil
	}

	// Direct assignment
	if srcVal.Type().AssignableTo(dstElem.Type()) {
		dstElem.Set(srcVal)
		return nil
	}

	// Type conversion
	if srcVal.Type().ConvertibleTo(dstElem.Type()) {
		dstElem.Set(srcVal.Convert(dstElem.Type()))
		return nil
	}

	return fmt.Errorf("cannot assign %T to %T", value, dst)
}
