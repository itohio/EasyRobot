package peg

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/graph"
)

// OpData holds metadata for operation nodes in the expression graph.
type OpData struct {
	Type        string // "literal", "wildcard", "field", "sequence", "choice", etc.
	Description string
	Value       interface{} // operation-specific data (e.g., []byte for literal, FieldSpec for field)
}

// ExpressionOp is the function signature for operations in the expression graph.
type ExpressionOp = graph.ExpressionOp[State, Decision]

// NewMatchLiteralOp creates an operation that matches an exact byte sequence.
func NewMatchLiteralOp(value []byte) ExpressionOp {
	return func(state State, _ map[int64]Decision) (Decision, bool) {
		packet := state.Packet()
		offset := state.Offset()
		end := offset + len(value)
		if end > len(packet) {
			state.SetDecision(DecisionContinue)
			return DecisionContinue, false // need more bytes
		}
		if !bytes.Equal(packet[offset:end], value) {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true // mismatch
		}
		state.SetOffset(end)
		state.SetDecision(DecisionContinue)
		return DecisionContinue, true // match, continue
	}
}

// NewMatchWildcardOp creates an operation that matches N arbitrary bytes.
func NewMatchWildcardOp(count int) ExpressionOp {
	return func(state State, _ map[int64]Decision) (Decision, bool) {
		packet := state.Packet()
		offset := state.Offset()
		end := offset + count
		if end > len(packet) {
			state.SetDecision(DecisionContinue)
			return DecisionContinue, false // need more bytes
		}
		state.SetOffset(end)
		state.SetDecision(DecisionContinue)
		return DecisionContinue, true // match, continue
	}
}

// NewDecodeFieldOp creates an operation that decodes a typed field and adds it to state.
func NewDecodeFieldOp(spec FieldSpec) ExpressionOp {
	return func(state State, _ map[int64]Decision) (Decision, bool) {
		packet := state.Packet()
		offset := state.Offset()
		packetLen := len(packet)

		switch spec.Type {
		case FieldStringPascal:
			if offset >= packetLen {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, false
			}
			length := int(packet[offset])
			end := offset + 1 + length
			if end > packetLen {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, false
			}
			value := string(packet[offset+1 : end])
			state.AddField(DecodedField{
				Name:   spec.Name,
				Offset: offset,
				Type:   spec.Type,
				Value:  value,
			})
			state.SetOffset(end)
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true

		case FieldStringC:
			i := offset
			for i < packetLen && packet[i] != 0 {
				i++
			}
			if i >= packetLen {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, false
			}
			value := string(packet[offset:i])
			state.AddField(DecodedField{
				Name:   spec.Name,
				Offset: offset,
				Type:   spec.Type,
				Value:  value,
			})
			state.SetOffset(i + 1)
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true

		case FieldStringFixed:
			if spec.Size <= 0 {
				state.SetDecision(DecisionDrop)
				return DecisionDrop, true
			}
			end := offset + spec.Size
			if end > packetLen {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, false
			}
			value := string(bytes.TrimRight(packet[offset:end], "\x00"))
			state.AddField(DecodedField{
				Name:   spec.Name,
				Offset: offset,
				Type:   spec.Type,
				Value:  value,
			})
			state.SetOffset(end)
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true

		case FieldVarintU, FieldVarintS:
			maxLen := 10
			if offset+maxLen > packetLen {
				maxLen = packetLen - offset
			}
			if maxLen <= 0 {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, false
			}
			var value interface{}
			var consumed int
			if spec.Type == FieldVarintU {
				val, nbytes := decodeVarintU(packet[offset:])
				if nbytes <= 0 {
					if maxLen > 0 && packetLen > offset && (packet[offset+maxLen-1]&0x80) != 0 {
						state.SetDecision(DecisionContinue)
						return DecisionContinue, false
					}
					if maxLen < 10 {
						state.SetDecision(DecisionContinue)
						return DecisionContinue, false
					}
					state.SetDecision(DecisionDrop)
					return DecisionDrop, true
				}
				value = val
				consumed = nbytes
			} else {
				val, nbytes := decodeVarintS(packet[offset:])
				if nbytes <= 0 {
					if maxLen > 0 && packetLen > offset && (packet[offset+maxLen-1]&0x80) != 0 {
						state.SetDecision(DecisionContinue)
						return DecisionContinue, false
					}
					if maxLen < 10 {
						state.SetDecision(DecisionContinue)
						return DecisionContinue, false
					}
					state.SetDecision(DecisionDrop)
					return DecisionDrop, true
				}
				value = val
				consumed = nbytes
			}
			state.AddField(DecodedField{
				Name:   spec.Name,
				Offset: offset,
				Value:  value,
				Type:   spec.Type,
			})
			state.SetOffset(offset + consumed)
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true

		default:
			size := spec.TotalBytes()
			if size <= 0 {
				state.SetDecision(DecisionDrop)
				return DecisionDrop, true
			}
			end := offset + size
			if end > packetLen {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, false
			}
			value, err := decodeField(packet[offset:end], spec)
			if err != nil {
				state.SetDecision(DecisionDrop)
				return DecisionDrop, true
			}
			state.AddField(DecodedField{
				Name:   spec.Name,
				Offset: offset,
				Value:  value,
				Type:   spec.Type,
			})
			if spec.Kind == FieldKindLength {
				switch v := value.(type) {
				case uint8:
					state.SetDeclaredLength(int(v))
				case uint16:
					state.SetDeclaredLength(int(v))
				case uint32:
					state.SetDeclaredLength(int(v))
				}
			}
			state.SetOffset(end)
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true
		}
	}
}

// NewSetMaxLengthOp creates an operation that sets the maximum packet length.
func NewSetMaxLengthOp(maxLen int) ExpressionOp {
	return func(state State, _ map[int64]Decision) (Decision, bool) {
		state.SetMaxLength(maxLen)
		state.SetDecision(DecisionContinue)
		return DecisionContinue, true
	}
}

// NewCheckLengthOp creates an operation that checks if packet length matches declared length.
func NewCheckLengthOp() ExpressionOp {
	return func(state State, _ map[int64]Decision) (Decision, bool) {
		declared := state.DeclaredLength()
		if declared <= 0 {
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true
		}
		current := state.CurrentLength()
		if current < declared {
			state.SetDecision(DecisionContinue)
			return DecisionContinue, false
		}
		if current > declared {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true
		}
		state.SetDecision(DecisionEmit)
		return DecisionEmit, true
	}
}

// NewCheckMaxLengthOp creates an operation that checks if packet exceeds max length.
func NewCheckMaxLengthOp() ExpressionOp {
	return func(state State, _ map[int64]Decision) (Decision, bool) {
		maxLen := state.MaxLength()
		if maxLen <= 0 {
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true
		}
		current := state.CurrentLength()
		if current > maxLen {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true
		}
		state.SetDecision(DecisionContinue)
		return DecisionContinue, true
	}
}

// NewSequenceOp creates an operation that evaluates children sequentially.
// Returns DecisionDrop if any child returns DecisionDrop.
// Returns DecisionKeep if any child returns DecisionKeep (and not all complete).
// Returns DecisionEmit if all children complete and last returns DecisionEmit.
func NewSequenceOp() ExpressionOp {
	return func(state State, childOutputs map[int64]Decision) (Decision, bool) {
		hasKeep := false
		hasDrop := false
		hasEmit := false
		for _, decision := range childOutputs {
			switch decision {
			case DecisionDrop:
				hasDrop = true
			case DecisionContinue:
				hasKeep = true
			case DecisionEmit:
				hasEmit = true
			}
		}
		if hasDrop {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true
		}
		if hasEmit && !hasKeep {
			state.SetDecision(DecisionEmit)
			return DecisionEmit, true
		}
		state.SetDecision(DecisionContinue)
		return DecisionContinue, !hasKeep // false if need more bytes
	}
}

// NewChoiceOp creates an operation that tries each child until one matches.
func NewChoiceOp() ExpressionOp {
	return func(state State, childOutputs map[int64]Decision) (Decision, bool) {
		hasKeep := false
		hasEmit := false
		for _, decision := range childOutputs {
			switch decision {
			case DecisionContinue:
				hasKeep = true
			case DecisionEmit:
				hasEmit = true
			case DecisionDrop:
				// ignore drops from other branches
			}
		}
		if hasEmit {
			state.SetDecision(DecisionEmit)
			return DecisionEmit, true
		}
		if hasKeep {
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true
		}
		state.SetDecision(DecisionDrop)
		return DecisionDrop, true
	}
}

// decodeField decodes a fixed-size field from bytes.
func decodeField(buf []byte, spec FieldSpec) (interface{}, error) {
	required := spec.TotalBytes()
	if required > 0 && len(buf) < required {
		return nil, fmt.Errorf("field requires %d bytes, have %d", required, len(buf))
	}
	switch spec.Type {
	case FieldUint8:
		return buf[0], nil
	case FieldInt8:
		return int8(buf[0]), nil
	case FieldUint16LE:
		return binary.LittleEndian.Uint16(buf), nil
	case FieldUint16BE:
		return binary.BigEndian.Uint16(buf), nil
	case FieldInt16LE:
		return int16(binary.LittleEndian.Uint16(buf)), nil
	case FieldInt16BE:
		return int16(binary.BigEndian.Uint16(buf)), nil
	case FieldUint32LE:
		return binary.LittleEndian.Uint32(buf), nil
	case FieldUint32BE:
		return binary.BigEndian.Uint32(buf), nil
	case FieldInt32LE:
		return int32(binary.LittleEndian.Uint32(buf)), nil
	case FieldInt32BE:
		return int32(binary.BigEndian.Uint32(buf)), nil
	case FieldUint64LE:
		return binary.LittleEndian.Uint64(buf), nil
	case FieldUint64BE:
		return binary.BigEndian.Uint64(buf), nil
	case FieldInt64LE:
		return int64(binary.LittleEndian.Uint64(buf)), nil
	case FieldInt64BE:
		return int64(binary.BigEndian.Uint64(buf)), nil
	case FieldFloat32:
		bits := binary.LittleEndian.Uint32(buf)
		return math.Float32frombits(bits), nil
	case FieldFloat64:
		bits := binary.LittleEndian.Uint64(buf)
		return math.Float64frombits(bits), nil
	default:
		return nil, fmt.Errorf("unsupported field type %d", spec.Type)
	}
}

// decodeVarintU decodes an unsigned varint (protobuf-style).
func decodeVarintU(data []byte) (uint64, int) {
	var result uint64
	var shift uint
	for i := 0; i < len(data) && i < 10; i++ {
		b := data[i]
		result |= uint64(b&0x7F) << shift
		if (b & 0x80) == 0 {
			return result, i + 1
		}
		shift += 7
		if shift >= 64 {
			return 0, 0
		}
	}
	return 0, 0
}

// decodeVarintS decodes a signed varint (protobuf-style zigzag encoding).
func decodeVarintS(data []byte) (int64, int) {
	val, n := decodeVarintU(data)
	if n == 0 {
		return 0, 0
	}
	// Zigzag decoding
	result := int64(val >> 1)
	if val&1 != 0 {
		result = ^result
	}
	return result, n
}
