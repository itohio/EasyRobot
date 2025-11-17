package graph

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"

	"google.golang.org/protobuf/proto"
)

// DataType represents the type of data stored
type DataType uint8

const (
	DataTypeProtobuf DataType = iota
	DataTypeBytes
	DataTypeString
	DataTypeInt
	DataTypeInt8
	DataTypeInt16
	DataTypeInt32
	DataTypeInt64
	DataTypeUint
	DataTypeUint8
	DataTypeUint16
	DataTypeUint32
	DataTypeUint64
	DataTypeFloat32
	DataTypeFloat64
	DataTypeArray
	DataTypeSlice
)

// serializeData serializes node or edge data to bytes.
// Supports: protobuf, bytes, strings, integers, floats, arrays, slices
// Returns payload bytes, data type, and optional protobuf type name.
func serializeData(data any) ([]byte, DataType, string, error) {
	if data == nil {
		return []byte{}, DataTypeBytes, "", nil
	}

	// Check for protobuf first
	if msg, ok := data.(proto.Message); ok {
		bytes, err := proto.Marshal(msg)
		if err != nil {
			return nil, 0, "", fmt.Errorf("failed to marshal protobuf: %w", err)
		}
		typeName := string(proto.MessageName(msg))
		if typeName == "" {
			typeName = reflect.TypeOf(msg).String()
		}
		return bytes, DataTypeProtobuf, typeName, nil
	}

	// Handle basic types
	switch v := data.(type) {
	case []byte:
		return v, DataTypeBytes, "", nil
	case string:
		return []byte(v), DataTypeString, "", nil
	case int:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, uint64(v))
		return buf, DataTypeInt, "", nil
	case int8:
		return []byte{byte(v)}, DataTypeInt8, "", nil
	case int16:
		buf := make([]byte, 2)
		binary.LittleEndian.PutUint16(buf, uint16(v))
		return buf, DataTypeInt16, "", nil
	case int32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(v))
		return buf, DataTypeInt32, "", nil
	case int64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, uint64(v))
		return buf, DataTypeInt64, "", nil
	case uint:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, uint64(v))
		return buf, DataTypeUint, "", nil
	case uint8:
		return []byte{v}, DataTypeUint8, "", nil
	case uint16:
		buf := make([]byte, 2)
		binary.LittleEndian.PutUint16(buf, v)
		return buf, DataTypeUint16, "", nil
	case uint32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, v)
		return buf, DataTypeUint32, "", nil
	case uint64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, v)
		return buf, DataTypeUint64, "", nil
	case float32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
		return buf, DataTypeFloat32, "", nil
	case float64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, math.Float64bits(v))
		return buf, DataTypeFloat64, "", nil
	}

	// Handle arrays and slices
	rv := reflect.ValueOf(data)
	switch rv.Kind() {
	case reflect.Array, reflect.Slice:
		return serializeArrayOrSlice(rv)
	}

	// Default: treat as bytes
	if bytes, ok := data.([]byte); ok {
		return bytes, DataTypeBytes, "", nil
	}

	return nil, 0, "", fmt.Errorf("unsupported data type: %T", data)
}

// serializeArrayOrSlice serializes arrays and slices
func serializeArrayOrSlice(rv reflect.Value) ([]byte, DataType, string, error) {
	length := rv.Len()
	if length == 0 {
		return []byte{}, DataTypeSlice, "", nil
	}

	// Get element type
	elemType := rv.Type().Elem()
	elemKind := elemType.Kind()

	// Serialize each element
	var result []byte
	for i := 0; i < length; i++ {
		elem := rv.Index(i).Interface()
		elemBytes, _, _, err := serializeData(elem)
		if err != nil {
			return nil, 0, "", fmt.Errorf("failed to serialize element %d: %w", i, err)
		}
		// Prepend element length for variable-length elements
		if needsLengthPrefix(elemKind) {
			lenBuf := make([]byte, 4)
			binary.LittleEndian.PutUint32(lenBuf, uint32(len(elemBytes)))
			result = append(result, lenBuf...)
		}
		result = append(result, elemBytes...)
	}

	// Prepend array/slice length
	lenBuf := make([]byte, 4)
	binary.LittleEndian.PutUint32(lenBuf, uint32(length))
	result = append(lenBuf, result...)

	if rv.Kind() == reflect.Array {
		return result, DataTypeArray, "", nil
	}
	return result, DataTypeSlice, "", nil
}

// needsLengthPrefix returns true if the element type needs a length prefix
func needsLengthPrefix(kind reflect.Kind) bool {
	switch kind {
	case reflect.String, reflect.Slice, reflect.Array:
		return true
	default:
		return false
	}
}

// deserializeData deserializes bytes back to the original data type
func deserializeData(data []byte, dataType DataType, typeName string, registry map[string]proto.Message) (any, error) {
	if len(data) == 0 {
		return nil, nil
	}

	switch dataType {
	case DataTypeProtobuf:
		if typeName == "" {
			return nil, fmt.Errorf("missing protobuf type information")
		}
		if registry == nil {
			return nil, fmt.Errorf("protobuf type %q not registered", typeName)
		}
		prototype, ok := registry[typeName]
		if !ok || prototype == nil {
			return nil, fmt.Errorf("protobuf type %q not registered", typeName)
		}
		msg := proto.Clone(prototype)
		if err := proto.Unmarshal(data, msg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal protobuf data: %w", err)
		}
		return msg, nil
	case DataTypeBytes:
		return data, nil
	case DataTypeString:
		return string(data), nil
	case DataTypeInt:
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for int: need 8 bytes, got %d", len(data))
		}
		return int(binary.LittleEndian.Uint64(data)), nil
	case DataTypeInt8:
		if len(data) < 1 {
			return nil, fmt.Errorf("insufficient data for int8: need 1 byte, got %d", len(data))
		}
		return int8(data[0]), nil
	case DataTypeInt16:
		if len(data) < 2 {
			return nil, fmt.Errorf("insufficient data for int16: need 2 bytes, got %d", len(data))
		}
		return int16(binary.LittleEndian.Uint16(data)), nil
	case DataTypeInt32:
		if len(data) < 4 {
			return nil, fmt.Errorf("insufficient data for int32: need 4 bytes, got %d", len(data))
		}
		return int32(binary.LittleEndian.Uint32(data)), nil
	case DataTypeInt64:
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for int64: need 8 bytes, got %d", len(data))
		}
		return int64(binary.LittleEndian.Uint64(data)), nil
	case DataTypeUint:
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for uint: need 8 bytes, got %d", len(data))
		}
		return uint(binary.LittleEndian.Uint64(data)), nil
	case DataTypeUint8:
		if len(data) < 1 {
			return nil, fmt.Errorf("insufficient data for uint8: need 1 byte, got %d", len(data))
		}
		return uint8(data[0]), nil
	case DataTypeUint16:
		if len(data) < 2 {
			return nil, fmt.Errorf("insufficient data for uint16: need 2 bytes, got %d", len(data))
		}
		return binary.LittleEndian.Uint16(data), nil
	case DataTypeUint32:
		if len(data) < 4 {
			return nil, fmt.Errorf("insufficient data for uint32: need 4 bytes, got %d", len(data))
		}
		return binary.LittleEndian.Uint32(data), nil
	case DataTypeUint64:
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for uint64: need 8 bytes, got %d", len(data))
		}
		return binary.LittleEndian.Uint64(data), nil
	case DataTypeFloat32:
		if len(data) < 4 {
			return nil, fmt.Errorf("insufficient data for float32: need 4 bytes, got %d", len(data))
		}
		bits := binary.LittleEndian.Uint32(data)
		return math.Float32frombits(bits), nil
	case DataTypeFloat64:
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for float64: need 8 bytes, got %d", len(data))
		}
		bits := binary.LittleEndian.Uint64(data)
		return math.Float64frombits(bits), nil
	case DataTypeArray, DataTypeSlice:
		return deserializeArrayOrSlice(data, dataType)
	default:
		return nil, fmt.Errorf("unknown data type: %d", dataType)
	}
}

// deserializeArrayOrSlice deserializes arrays and slices
func deserializeArrayOrSlice(data []byte, dataType DataType) (any, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("insufficient data for array/slice length")
	}
	length := int(binary.LittleEndian.Uint32(data[:4]))
	if length == 0 {
		if dataType == DataTypeArray {
			return []any{}, nil
		}
		return []any{}, nil
	}

	// For now, return as []any - caller can type assert to specific type
	result := make([]any, 0, length)
	offset := 4
	for i := 0; i < length && offset < len(data); i++ {
		// Try to deserialize as int64 first (most common)
		if offset+8 <= len(data) {
			val := int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
			result = append(result, val)
			offset += 8
		} else {
			// Fallback: return remaining bytes
			result = append(result, data[offset:])
			break
		}
	}

	return result, nil
}
