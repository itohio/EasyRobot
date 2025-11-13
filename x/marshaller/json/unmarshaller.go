package json

import (
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Unmarshaller implements JSON-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new JSON unmarshaller.
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
	return "json"
}

// Unmarshal decodes value from JSON format.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	localOpts := u.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	decoder := json.NewDecoder(r)

	// Decode jsonValue
	var jv jsonValue
	if err := decoder.Decode(&jv); err != nil {
		return types.NewError("unmarshal", "json", "decoding", err)
	}

	// Convert jsonValue to actual value
	value, err := u.jsonToValue(&jv, localOpts)
	if err != nil {
		return types.NewError("unmarshal", "json", "conversion", err)
	}

	// Assign to dst
	if err := u.assignToDst(dst, value); err != nil {
		return types.NewError("unmarshal", "json", "assignment", err)
	}

	return nil
}

func (u *Unmarshaller) jsonToValue(jv *jsonValue, opts types.Options) (any, error) {
	if jv == nil {
		return nil, fmt.Errorf("nil jsonValue")
	}

	// Get tensor factory from options, default to tensor.New wrapped
	tensorFactory := opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dtype types.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	switch jv.Kind {
	case "tensor":
		if jv.Tensor == nil {
			return nil, fmt.Errorf("nil tensor in jsonValue")
		}
		return jsonToTensor(jv.Tensor, opts.DestinationType, tensorFactory)

	case "layer":
		if jv.Layer == nil {
			return nil, fmt.Errorf("nil layer in jsonValue")
		}
		// For layers, we cannot fully reconstruct them without type information
		return *jv.Layer, nil

	case "model":
		if jv.Model == nil {
			return nil, fmt.Errorf("nil model in jsonValue")
		}
		// For models, similar to layers, we cannot fully reconstruct them
		return *jv.Model, nil

	case "slice":
		// For slices, JSON already decoded the data, but we need to convert types
		return convertSliceData(jv.SliceData, jv.SliceType)

	case "generic":
		return nil, fmt.Errorf("cannot reconstruct generic value")

	default:
		return nil, fmt.Errorf("unknown jsonValue kind: %s", jv.Kind)
	}
}

func jsonToTensor(jt *jsonTensor, destType types.DataType, tensorFactory func(types.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if jt == nil || len(jt.Shape) == 0 {
		return nil, fmt.Errorf("invalid tensor data")
	}

	dtype := stringToDtype(jt.DataType)
	shape := types.Shape(jt.Shape)

	// Create tensor
	var t types.Tensor
	if destType != 0 {
		// Create with destination type for conversion
		t = tensorFactory(destType, shape)
	} else {
		t = tensorFactory(dtype, shape)
	}

	// Copy data from JSON - need to handle type conversion
	// JSON decodes numbers as float64 by default
	if dataSlice, ok := jt.Data.([]any); ok {
		for i, v := range dataSlice {
			if i >= t.Size() {
				break
			}
			var val float64
			switch v := v.(type) {
			case float64:
				val = v
			case float32:
				val = float64(v)
			case int:
				val = float64(v)
			case int64:
				val = float64(v)
			default:
				return nil, fmt.Errorf("unsupported data type in JSON: %T", v)
			}
			t.SetAt(val, i)
		}
	}

	return t, nil
}

func convertSliceData(data any, sliceType string) (any, error) {
	// JSON decodes numeric arrays as []any, we need to convert to proper type
	if data == nil {
		return nil, fmt.Errorf("nil slice data")
	}

	anySlice, ok := data.([]any)
	if !ok {
		// Data is already the correct type
		return data, nil
	}

	// Convert based on slice type
	switch sliceType {
	case "[]float32":
		result := make([]float32, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = float32(f)
			}
		}
		return result, nil
	case "[]float64":
		result := make([]float64, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = f
			}
		}
		return result, nil
	case "[]int":
		result := make([]int, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = int(f)
			}
		}
		return result, nil
	case "[]int64":
		result := make([]int64, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = int64(f)
			}
		}
		return result, nil
	case "[]int32":
		result := make([]int32, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = int32(f)
			}
		}
		return result, nil
	default:
		return data, nil
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
