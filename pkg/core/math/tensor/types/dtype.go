package types

// DataType represents the underlying element type stored by a tensor.
type DataType uint8

const (
	DT_UNKNOWN DataType = iota
	DTFP32              // DTFP32 represents 32-bit floating point tensors (default).
	DTFP64              // 64-bit floating point tensors
	DTFP16              // 16-bit floating point tensors
	DTINT48             // 4-bit integer tensors unpacked into 8bit
	DTINT8              // 8-bit integer tensors
	DTINT16             // 16-bit integer tensors
)

// DataElementType is the type constraint for the data elements in the tensor.
type DataElementType interface {
	~float64 | ~float32 | ~int16 | ~int8
}

func TypeFromData(v any) DataType {
	switch any(v).(type) {
	case float64:
		return DTFP64
	case float32:
		return DTFP32
	case int16:
		return DTINT16
	case int8:
		return DTINT8
	case []float64:
		return DTFP64
	case []float32:
		return DTFP32
	case []int16:
		return DTINT16
	case []int8:
		return DTINT8
	default:
		return DT_UNKNOWN
	}
}

func MakeTensorData(dt DataType, size int) any {
	switch dt {
	case DTFP32:
		return make([]float32, size)
	case DTFP64:
		return make([]float64, size)
	case DTINT16:
		return make([]int16, size)
	case DTINT8:
		return make([]int8, size)
	case DTINT48:
		return make([]int8, size)
	default:
		return nil
	}
}

func CloneTensorDataTo(dst DataType, data any) any {
	if data == nil {
		return nil
	}
	size := 0
	switch d := data.(type) {
	case []float32:
		size = len(d)
	case []float64:
		size = len(d)
	case []int16:
		size = len(d)
	case []int8:
		size = len(d)
	default:
		return nil
	}

	newData := MakeTensorData(dst, size)
	if newData == nil {
		return nil
	}
	return CopyTensorData(dst, newData, data)
}

func CloneTensorData(data any) any {
	if data == nil {
		return nil
	}
	return CloneTensorDataTo(TypeFromData(data), data)
}

func CopyTensorData(dst DataType, dstData, srcData any) any {
	if srcData == nil || dstData == nil {
		return nil
	}

	switch dst {
	case DTFP32:
		dstSlice, ok := dstData.([]float32)
		if !ok {
			return nil
		}
		switch src := srcData.(type) {
		case []float32:
			copy(dstSlice, src)
		case []float64:
			for i := range dstSlice {
				dstSlice[i] = float32(src[i])
			}
		case []int16:
			for i := range dstSlice {
				dstSlice[i] = float32(src[i])
			}
		case []int8:
			for i := range dstSlice {
				dstSlice[i] = float32(src[i])
			}
		default:
			return nil
		}
		return dstSlice
	case DTFP64:
		dstSlice, ok := dstData.([]float64)
		if !ok {
			return nil
		}
		switch src := srcData.(type) {
		case []float32:
			for i := range dstSlice {
				dstSlice[i] = float64(src[i])
			}
		case []float64:
			copy(dstSlice, src)
		case []int16:
			for i := range dstSlice {
				dstSlice[i] = float64(src[i])
			}
		case []int8:
			for i := range dstSlice {
				dstSlice[i] = float64(src[i])
			}
		default:
			return nil
		}
		return dstSlice
	case DTINT16:
		dstSlice, ok := dstData.([]int16)
		if !ok {
			return nil
		}
		switch src := srcData.(type) {
		case []float32:
			for i := range dstSlice {
				dstSlice[i] = int16(src[i])
			}
		case []float64:
			for i := range dstSlice {
				dstSlice[i] = int16(src[i])
			}
		case []int16:
			copy(dstSlice, src)
		case []int8:
			for i := range dstSlice {
				dstSlice[i] = int16(src[i])
			}
		default:
			return nil
		}
		return dstSlice
	case DTINT8, DTINT48:
		dstSlice, ok := dstData.([]int8)
		if !ok {
			return nil
		}
		switch src := srcData.(type) {
		case []float32:
			for i := range dstSlice {
				dstSlice[i] = int8(src[i])
			}
		case []float64:
			for i := range dstSlice {
				dstSlice[i] = int8(src[i])
			}
		case []int16:
			for i := range dstSlice {
				dstSlice[i] = int8(src[i])
			}
		case []int8:
			copy(dstSlice, src)
		default:
			return nil
		}
		return dstSlice
	default:
		return nil
	}
}

// Helper functions to work with interface tensors
func GetTensorData[T any](t Tensor) T {
	if t == nil {
		var zero T
		return zero
	}
	data := t.Data()
	if data == nil {
		var zero T
		return zero
	}
	return data.(T)
}
