package yaml

import (
	"fmt"
	"io"
	"reflect"

	"gopkg.in/yaml.v3"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Unmarshaller implements YAML-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new YAML unmarshaller.
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
	return "yaml"
}

// Unmarshal decodes value from YAML format.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	localOpts := u.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	decoder := yaml.NewDecoder(r)

	// Decode yamlValue
	var yv yamlValue
	if err := decoder.Decode(&yv); err != nil {
		return types.NewError("unmarshal", "yaml", "decoding", err)
	}

	// Convert yamlValue to actual value
	value, err := u.yamlToValue(&yv, localOpts)
	if err != nil {
		return types.NewError("unmarshal", "yaml", "conversion", err)
	}

	// Assign to dst
	if err := u.assignToDst(dst, value); err != nil {
		return types.NewError("unmarshal", "yaml", "assignment", err)
	}

	return nil
}

func (u *Unmarshaller) yamlToValue(yv *yamlValue, opts types.Options) (any, error) {
	if yv == nil {
		return nil, fmt.Errorf("nil yamlValue")
	}

	// Get tensor factory from options, default to tensor.New wrapped
	tensorFactory := opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dtype types.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	switch yv.Kind {
	case "tensor":
		if yv.Tensor == nil {
			return nil, fmt.Errorf("nil tensor in yamlValue")
		}
		return yamlToTensor(yv.Tensor, opts.DestinationType, tensorFactory)

	case "layer":
		if yv.Layer == nil {
			return nil, fmt.Errorf("nil layer in yamlValue")
		}
		return *yv.Layer, nil

	case "model":
		if yv.Model == nil {
			return nil, fmt.Errorf("nil model in yamlValue")
		}
		return *yv.Model, nil

	case "slice":
		// For slices, YAML already decoded the data, but we need to convert types
		return convertSliceData(yv.SliceData, yv.SliceType)

	case "generic":
		return nil, fmt.Errorf("cannot reconstruct generic value")

	default:
		return nil, fmt.Errorf("unknown yamlValue kind: %s", yv.Kind)
	}
}

func yamlToTensor(yt *yamlTensor, destType types.DataType, tensorFactory func(types.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if yt == nil || len(yt.Shape) == 0 {
		return nil, fmt.Errorf("invalid tensor data")
	}

	dtype := stringToDtype(yt.DataType)
	shape := types.Shape(yt.Shape)

	// Create tensor
	var t types.Tensor
	if destType != 0 {
		t = tensorFactory(destType, shape)
	} else {
		t = tensorFactory(dtype, shape)
	}

	// Copy data from YAML
	if dataSlice, ok := yt.Data.([]any); ok {
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
				return nil, fmt.Errorf("unsupported data type in YAML: %T", v)
			}
			t.SetAt(val, i)
		}
	}

	return t, nil
}

func convertSliceData(data any, sliceType string) (any, error) {
	// YAML decodes numeric arrays as []any, we need to convert to proper type
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
			switch v := v.(type) {
			case float64:
				result[i] = float32(v)
			case int:
				result[i] = float32(v)
			}
		}
		return result, nil
	case "[]float64":
		result := make([]float64, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case float64:
				result[i] = v
			case int:
				result[i] = float64(v)
			}
		}
		return result, nil
	case "[]int":
		result := make([]int, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case int:
				result[i] = v
			case float64:
				result[i] = int(v)
			}
		}
		return result, nil
	case "[]int64":
		result := make([]int64, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case int64:
				result[i] = v
			case int:
				result[i] = int64(v)
			case float64:
				result[i] = int64(v)
			}
		}
		return result, nil
	case "[]int32":
		result := make([]int32, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case int32:
				result[i] = v
			case int:
				result[i] = int32(v)
			case float64:
				result[i] = int32(v)
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

	dstVal := reflect.ValueOf(dst)
	if dstVal.Kind() != reflect.Ptr {
		return fmt.Errorf("dst must be a pointer, got %s", dstVal.Kind())
	}

	dstElem := dstVal.Elem()
	if !dstElem.CanSet() {
		return fmt.Errorf("dst element cannot be set")
	}

	valueVal := reflect.ValueOf(value)
	if !valueVal.Type().AssignableTo(dstElem.Type()) {
		return fmt.Errorf("cannot assign %s to %s", valueVal.Type(), dstElem.Type())
	}

	dstElem.Set(valueVal)
	return nil
}

