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
	DataTypeComplex64
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
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(v))
		return append([]byte(nil), buf[:]...), DataTypeInt, "", nil
	case int8:
		return []byte{byte(v)}, DataTypeInt8, "", nil
	case int16:
		var buf [2]byte
		binary.LittleEndian.PutUint16(buf[:], uint16(v))
		return append([]byte(nil), buf[:]...), DataTypeInt16, "", nil
	case int32:
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], uint32(v))
		return append([]byte(nil), buf[:]...), DataTypeInt32, "", nil
	case int64:
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(v))
		return append([]byte(nil), buf[:]...), DataTypeInt64, "", nil
	case uint:
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(v))
		return append([]byte(nil), buf[:]...), DataTypeUint, "", nil
	case uint8:
		return []byte{v}, DataTypeUint8, "", nil
	case uint16:
		var buf [2]byte
		binary.LittleEndian.PutUint16(buf[:], v)
		return append([]byte(nil), buf[:]...), DataTypeUint16, "", nil
	case uint32:
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], v)
		return append([]byte(nil), buf[:]...), DataTypeUint32, "", nil
	case uint64:
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], v)
		return append([]byte(nil), buf[:]...), DataTypeUint64, "", nil
	case float32:
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		return append([]byte(nil), buf[:]...), DataTypeFloat32, "", nil
	case float64:
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		return append([]byte(nil), buf[:]...), DataTypeFloat64, "", nil
	case complex64:
		// Store complex64 as two float32 values: real, imag
		var buf [8]byte
		realBits := math.Float32bits(real(v))
		imagBits := math.Float32bits(imag(v))
		binary.LittleEndian.PutUint32(buf[0:4], realBits)
		binary.LittleEndian.PutUint32(buf[4:8], imagBits)
		return append([]byte(nil), buf[:]...), DataTypeComplex64, "", nil
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
// Format: [4 bytes: length][for each element: 1 byte type][4 bytes: typeNameLen][typeName bytes][4 bytes: elemLen if needed][element bytes]
func serializeArrayOrSlice(rv reflect.Value) ([]byte, DataType, string, error) {
	length := rv.Len()
	if length == 0 {
		return []byte{}, DataTypeSlice, "", nil
	}

	// Get element type
	elemType := rv.Type().Elem()
	elemKind := elemType.Kind()

	// Serialize each element with type information
	var result []byte
	for i := 0; i < length; i++ {
		elem := rv.Index(i).Interface()
		elemBytes, elemDataType, elemTypeName, err := serializeData(elem)
		if err != nil {
			return nil, 0, "", fmt.Errorf("failed to serialize element %d: %w", i, err)
		}

		// Store element type (1 byte)
		result = append(result, byte(elemDataType))

		// Store type name length and type name (for protobuf and future extensibility)
		typeNameBytes := []byte(elemTypeName)
		typeNameLen := uint16(len(typeNameBytes))
		var lenBuf2 [2]byte
		binary.LittleEndian.PutUint16(lenBuf2[:], typeNameLen)
		result = append(result, lenBuf2[:]...)
		if typeNameLen > 0 {
			result = append(result, typeNameBytes...)
		}

		// Prepend element length for variable-length elements
		if needsLengthPrefix(elemKind) {
			var lenBuf4 [4]byte
			binary.LittleEndian.PutUint32(lenBuf4[:], uint32(len(elemBytes)))
			result = append(result, lenBuf4[:]...)
		}
		result = append(result, elemBytes...)
	}

	// Prepend array/slice length
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(length))
	result = append(lenBuf[:], result...)

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
	case DataTypeComplex64:
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for complex64: need 8 bytes, got %d", len(data))
		}
		realBits := binary.LittleEndian.Uint32(data[0:4])
		imagBits := binary.LittleEndian.Uint32(data[4:8])
		realVal := math.Float32frombits(realBits)
		imagVal := math.Float32frombits(imagBits)
		return complex(realVal, imagVal), nil
	case DataTypeArray, DataTypeSlice:
		return deserializeArrayOrSlice(data, dataType, registry)
	default:
		return nil, fmt.Errorf("unknown data type: %d", dataType)
	}
}

// deserializeArrayOrSlice deserializes arrays and slices
// Format: [4 bytes: length][for each element: 1 byte type][2 bytes: typeNameLen][typeName bytes][4 bytes: elemLen if needed][element bytes]
func deserializeArrayOrSlice(data []byte, dataType DataType, registry map[string]proto.Message) (any, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("insufficient data for array/slice length")
	}
	length := int(binary.LittleEndian.Uint32(data[:4]))
	if length == 0 {
		return []any{}, nil
	}

	result := make([]any, 0, length)
	offset := 4
	for i := 0; i < length && offset < len(data); i++ {
		// Read element type
		if offset >= len(data) {
			return nil, fmt.Errorf("truncated data at element %d: missing type", i)
		}
		elemDataType := DataType(data[offset])
		offset++

		// Read type name length and type name
		if offset+2 > len(data) {
			return nil, fmt.Errorf("truncated data at element %d: missing type name length", i)
		}
		typeNameLen := int(binary.LittleEndian.Uint16(data[offset : offset+2]))
		offset += 2

		var elemTypeName string
		if typeNameLen > 0 {
			if offset+typeNameLen > len(data) {
				return nil, fmt.Errorf("truncated data at element %d: missing type name", i)
			}
			elemTypeName = string(data[offset : offset+typeNameLen])
			offset += typeNameLen
		}

		// Determine if element needs length prefix
		needsLenPrefix := elemDataType == DataTypeProtobuf || elemDataType == DataTypeBytes || elemDataType == DataTypeString || elemDataType == DataTypeArray || elemDataType == DataTypeSlice

		var elemLen int
		if needsLenPrefix {
			if offset+4 > len(data) {
				return nil, fmt.Errorf("truncated data at element %d: missing element length", i)
			}
			elemLen = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4
		} else {
			// Fixed-size types
			elemLen = getFixedSize(elemDataType)
			if elemLen == 0 {
				return nil, fmt.Errorf("unknown fixed size for data type %d", elemDataType)
			}
		}

		// Read element bytes
		if offset+elemLen > len(data) {
			return nil, fmt.Errorf("truncated data at element %d: need %d bytes, have %d", i, elemLen, len(data)-offset)
		}
		elemBytes := data[offset : offset+elemLen]
		offset += elemLen

		// Deserialize element
		elem, err := deserializeData(elemBytes, elemDataType, elemTypeName, registry)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize element %d: %w", i, err)
		}
		result = append(result, elem)
	}

	return result, nil
}

// getFixedSize returns the fixed byte size for a data type, or 0 if variable-length
func getFixedSize(dataType DataType) int {
	switch dataType {
	case DataTypeInt8, DataTypeUint8:
		return 1
	case DataTypeInt16, DataTypeUint16:
		return 2
	case DataTypeInt32, DataTypeUint32, DataTypeFloat32:
		return 4
	case DataTypeInt64, DataTypeUint64, DataTypeInt, DataTypeUint, DataTypeFloat64, DataTypeComplex64:
		return 8
	default:
		return 0 // Variable-length
	}
}
