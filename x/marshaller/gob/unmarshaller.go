package gob

import (
	"encoding/gob"
	"fmt"
	"io"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Unmarshaller implements gob-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new gob unmarshaller.
func NewUnmarshaller(opts ...types.Option) types.Unmarshaller {
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
	return "gob"
}

// Unmarshal decodes value from gob format.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	localOpts := u.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	decoder := gob.NewDecoder(r)

	// Decode gobValue
	var gv gobValue
	if err := decoder.Decode(&gv); err != nil {
		return types.NewError("unmarshal", "gob", "decoding", err)
	}

	// Convert gobValue to actual value
	value, err := u.gobToValue(&gv, localOpts)
	if err != nil {
		return types.NewError("unmarshal", "gob", "conversion", err)
	}

	// Assign to dst
	if err := u.assignToDst(dst, value); err != nil {
		return types.NewError("unmarshal", "gob", "assignment", err)
	}

	return nil
}

func (u *Unmarshaller) gobToValue(gv *gobValue, opts types.Options) (any, error) {
	if gv == nil {
		return nil, fmt.Errorf("nil gobValue")
	}

	// Get tensor factory from options, default to tensor.New wrapped
	tensorFactory := opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dtype types.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	switch gv.Kind {
	case "tensor":
		if gv.Tensor == nil {
			return nil, fmt.Errorf("nil tensor in gobValue")
		}
		// Check if type conversion is requested
		if opts.DestinationType != 0 {
			return gobToTensorWithConversion(*gv.Tensor, opts.DestinationType, tensorFactory)
		}
		return gobToTensor(*gv.Tensor, tensorFactory), nil

	case "layer":
		if gv.Layer == nil {
			return nil, fmt.Errorf("nil layer in gobValue")
		}
		// For layers, we cannot fully reconstruct them without type information
		// This would require a layer factory based on the Type field
		// For now, we'll return the gobLayer itself
		// In a full implementation, you'd need a registry of layer constructors
		return *gv.Layer, nil

	case "model":
		if gv.Model == nil {
			return nil, fmt.Errorf("nil model in gobValue")
		}
		// For models, similar to layers, we cannot fully reconstruct them
		// This would require a model factory based on the Type field
		// For now, we'll return the gobModel itself
		return *gv.Model, nil

	case "slice":
		// For slices, gob already decoded the data
		return gv.SliceData, nil

	case "generic":
		// For generic values, we cannot reconstruct without more information
		return nil, fmt.Errorf("cannot reconstruct generic value")

	default:
		return nil, fmt.Errorf("unknown gobValue kind: %s", gv.Kind)
	}
}

func (u *Unmarshaller) assignToDst(dst any, value any) error {
	if dst == nil {
		return fmt.Errorf("dst is nil")
	}

	// dst should be a pointer
	dstVal := reflect.ValueOf(dst)
	if dstVal.Kind() != reflect.Ptr {
		return fmt.Errorf("dst must be a pointer, got %s", dstVal.Kind())
	}

	// Get the element the pointer points to
	dstElem := dstVal.Elem()
	if !dstElem.CanSet() {
		return fmt.Errorf("dst element cannot be set")
	}

	// Set the value
	valueVal := reflect.ValueOf(value)
	if !valueVal.Type().AssignableTo(dstElem.Type()) {
		return fmt.Errorf("cannot assign %s to %s", valueVal.Type(), dstElem.Type())
	}

	dstElem.Set(valueVal)
	return nil
}
