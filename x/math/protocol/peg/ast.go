package peg

import (
	"bytes"
	"fmt"
)

type evalStatus int

const (
	statusMatch evalStatus = iota
	statusNeedMore
)

type node interface {
	eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error)
}

type sequenceNode struct {
	children []node
}

func (n *sequenceNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	state.SetOffset(offset)
	curr := offset
	for i, child := range n.children {
		// Use state's current offset (may have been changed by offset jumps)
		curr = state.Offset()

		// Special handling for wildcard nodes: if there are more children after this wildcard,
		// don't let it consume all bytes - it should leave room for subsequent fields
		if wc, ok := child.(*wildcardNode); ok && wc.count < 0 && i < len(n.children)-1 {
			// Wildcard with no count followed by more nodes
			// Calculate how many bytes remain and ensure we leave space for ALL remaining fields
			available := len(data) - curr

			// Calculate total size of all remaining fields after this wildcard
			totalRemainingSize := 0
			for j := i + 1; j < len(n.children); j++ {
				if fieldNode, ok := n.children[j].(*fieldNode); ok {
					fieldSize := fieldNode.spec.TotalBytes()
					if fieldSize > 0 {
						totalRemainingSize += fieldSize
					} else {
						// Variable-length field - can't calculate ahead, so don't restrict wildcard
						totalRemainingSize = 0
						break
					}
				} else if literalNode, ok := n.children[j].(*literalNode); ok {
					totalRemainingSize += len(literalNode.value)
				} else if _, ok := n.children[j].(*offsetNode); ok {
					// Offset jumps don't consume bytes, skip
					continue
				} else {
					// Unknown node type - can't calculate ahead, so don't restrict wildcard
					totalRemainingSize = 0
					break
				}
			}

			if totalRemainingSize > 0 {
				// Wildcard * when followed by fields should skip at least 1 byte
				// We need: 1 byte (wildcard) + totalRemainingSize bytes (fields) <= available
				minNeeded := 1 + totalRemainingSize
				if available < minNeeded {
					// Not enough bytes to skip 1 byte and read all fields
					// Set wildcard to require exactly minNeeded bytes - this will cause it
					// to return statusNeedMore when available < minNeeded
					wc.count = minNeeded
					// Don't continue - let the wildcard eval() return statusNeedMore
				} else {
					// Leave room for all remaining fields - wildcard should match available - totalRemainingSize
					// But ensure we skip at least 1 byte
					wc.count = available - totalRemainingSize
					if wc.count < 1 {
						wc.count = 1
					}
				}
			}
		}

		status, next, err := child.eval(ctx, data, curr, state)
		if err != nil {
			return statusMatch, 0, err
		}
		if status == statusNeedMore {
			return statusNeedMore, 0, nil
		}
		// Update state offset to next position
		state.SetOffset(next)
		curr = next

		// Restore wildcard count if we modified it
		if wc, ok := child.(*wildcardNode); ok && wc.count >= 0 {
			wc.count = -1
		}
	}
	return statusMatch, curr, nil
}

type choiceNode struct {
	branches []node
}

func (n *choiceNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	var needMore bool
	var lastErr error
	for _, branch := range n.branches {
		cloned := state.Clone()
		// Save original offset for backtracking
		originalOffset := cloned.Offset()
		cloned.SetOffset(offset)
		status, next, err := branch.eval(ctx, data, offset, cloned)
		if err == nil && status == statusMatch {
			// Restore offset to group start for backtracking
			cloned.SetOffset(originalOffset)
			state.Merge(cloned)
			return statusMatch, next, nil
		}
		if err != nil {
			lastErr = err
			continue
		}
		if status == statusNeedMore {
			needMore = true
		}
	}
	if needMore && lastErr == nil {
		return statusNeedMore, 0, nil
	}
	if lastErr != nil {
		return statusMatch, 0, lastErr
	}
	return statusMatch, 0, fmt.Errorf("no branch matched")
}

type literalNode struct {
	value []byte
}

func (n *literalNode) eval(_ *evalContext, data []byte, offset int, _ State) (evalStatus, int, error) {
	end := offset + len(n.value)
	if end > len(data) {
		return statusNeedMore, 0, nil
	}
	if !bytes.Equal(data[offset:end], n.value) {
		return statusMatch, 0, fmt.Errorf("literal mismatch at offset %d", offset)
	}
	return statusMatch, end, nil
}

type wildcardNode struct {
	count int
}

type lookaheadNode struct {
	child node
}

func (n *lookaheadNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	saved := state.Offset()
	state.SetOffset(offset)
	status, _, err := n.child.eval(ctx, data, offset, state)
	if err != nil {
		return statusMatch, 0, err
	}
	if status == statusNeedMore {
		return statusNeedMore, 0, nil
	}
	state.SetOffset(saved)
	return statusMatch, saved, nil
}

func (n *wildcardNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	var end int
	if n.count < 0 {
		// Negative count means "until MaxLength" - match all available bytes up to MaxLength
		maxLen := state.MaxLength()
		available := len(data) - offset

		if maxLen > 0 {
			// Match all available bytes, but don't exceed MaxLength
			maxAvailable := maxLen - offset
			if maxAvailable < 0 {
				maxAvailable = 0
			}
			if available > maxAvailable {
				available = maxAvailable
			}
		}

		// If we're in a sequence and there might be more nodes after this wildcard,
		// we need to leave at least some bytes. However, we don't know what comes next.
		// For now, match what's available but return needMore if we consumed everything
		// This will allow the next field to request more bytes if needed
		end = offset + available

		// Always return statusMatch for wildcard with * (no number) - it matches what's available
		return statusMatch, end, nil
	} else {
		end = offset + n.count
		if end > len(data) {
			return statusNeedMore, 0, nil
		}
	}
	return statusMatch, end, nil
}

type fieldNode struct {
	spec FieldSpec
}

func (n *fieldNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	// Use state's current offset if it differs (e.g., after offset jump)
	if state.Offset() != offset {
		offset = state.Offset()
	}
	switch n.spec.Type {
	case FieldStringPascal:
		if offset >= len(data) {
			return statusNeedMore, 0, nil
		}
		length := int(data[offset])
		end := offset + 1 + length
		if end > len(data) {
			return statusNeedMore, 0, nil
		}
		value := string(data[offset+1 : end])
		state.AddField(DecodedField{Name: n.spec.Name, Offset: offset, Type: n.spec.Type, Value: value})
		return statusMatch, end, nil
	case FieldStringC:
		i := offset
		for i < len(data) && data[i] != 0 {
			i++
		}
		if i >= len(data) {
			return statusNeedMore, 0, nil
		}
		value := string(data[offset:i])
		state.AddField(DecodedField{Name: n.spec.Name, Offset: offset, Type: n.spec.Type, Value: value})
		return statusMatch, i + 1, nil
	case FieldVarintU, FieldVarintS:
		// Varints are variable length, need to check if we have enough bytes
		// Try to decode with up to 10 bytes (max for uint64)
		maxLen := 10
		if offset+maxLen > len(data) {
			maxLen = len(data) - offset
		}
		if maxLen <= 0 {
			return statusNeedMore, 0, nil
		}
		var value interface{}
		var consumed int
		if n.spec.Type == FieldVarintU {
			val, nbytes := decodeVarintU(data[offset:])
			if nbytes <= 0 {
				// Incomplete varint - check if last byte has continuation bit
				if maxLen > 0 && len(data) > offset && (data[offset+maxLen-1]&0x80) != 0 {
					return statusNeedMore, 0, nil
				}
				// If we have less than 10 bytes and haven't hit end of data, might need more
				if maxLen < 10 {
					return statusNeedMore, 0, nil
				}
				return statusMatch, 0, fmt.Errorf("invalid varint")
			}
			value = val
			consumed = nbytes
		} else {
			val, nbytes := decodeVarintS(data[offset:])
			if nbytes <= 0 {
				// Incomplete varint - check if last byte has continuation bit
				if maxLen > 0 && len(data) > offset && (data[offset+maxLen-1]&0x80) != 0 {
					return statusNeedMore, 0, nil
				}
				if maxLen < 10 {
					return statusNeedMore, 0, nil
				}
				return statusMatch, 0, fmt.Errorf("invalid varint")
			}
			value = val
			consumed = nbytes
		}
		state.AddField(DecodedField{
			Name:   n.spec.Name,
			Offset: offset,
			Value:  value,
			Type:   n.spec.Type,
		})
		return statusMatch, offset + consumed, nil
	default:
		size := n.spec.TotalBytes()
		if size <= 0 {
			return statusMatch, 0, fmt.Errorf("field %s has invalid size", n.spec.Name)
		}
		end := offset + size
		if end > len(data) {
			return statusNeedMore, 0, nil
		}
		value, err := decodeField(data[offset:end], n.spec)
		if err != nil {
			return statusMatch, 0, err
		}
		state.AddField(DecodedField{
			Name:   n.spec.Name,
			Offset: offset,
			Value:  value,
			Type:   n.spec.Type,
		})
		if n.spec.Kind == FieldKindLength {
			switch v := value.(type) {
			case uint8:
				state.SetDeclaredLength(int(v))
			case uint16:
				state.SetDeclaredLength(int(v))
			}
		}
		return statusMatch, end, nil
	}
}

// decodeField is defined in ops.go

type offsetNode struct {
	offset   int
	relative bool
}

func (n *offsetNode) eval(_ *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	var targetOffset int
	if n.relative {
		targetOffset = offset + n.offset
	} else {
		targetOffset = n.offset
	}

	if targetOffset < 0 {
		return statusMatch, 0, fmt.Errorf("invalid negative offset %d", targetOffset)
	}

	// Check MaxLength
	maxLen := state.MaxLength()
	if maxLen > 0 && targetOffset > maxLen {
		// Offset exceeds MaxLength - drop the packet
		return statusMatch, 0, fmt.Errorf("offset %d exceeds MaxLength %d", targetOffset, maxLen)
	}

	if targetOffset > len(data) {
		return statusNeedMore, 0, nil
	}

	// Set state offset to the target offset
	state.SetOffset(targetOffset)
	return statusMatch, targetOffset, nil
}

type skipUntilNode struct {
	pattern []byte
}

func (n *skipUntilNode) eval(_ *evalContext, data []byte, offset int, _ State) (evalStatus, int, error) {
	if len(n.pattern) == 0 {
		return statusMatch, offset, nil
	}

	// Search for pattern starting from offset
	for i := offset; i <= len(data)-len(n.pattern); i++ {
		if bytes.Equal(data[i:i+len(n.pattern)], n.pattern) {
			return statusMatch, i + len(n.pattern), nil
		}
	}

	// Pattern not found yet, need more bytes
	return statusNeedMore, 0, nil
}

type arrayWithStarNode struct {
	spec     FieldSpec
	baseName string
}

func (n *arrayWithStarNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	// Read until exhausted - calculate remaining bytes from declared length or max length
	fieldSize := n.spec.TotalBytes()
	if fieldSize == 0 {
		// Variable length field (varint, string) - can't use star array easily
		return statusMatch, 0, fmt.Errorf("array with star not supported for variable-length fields")
	}

	// Calculate how many bytes remain after this array
	// We need to know the total packet length and what comes after this array
	declaredLen := state.DeclaredLength()
	maxLen := state.MaxLength()

	var availableBytes int
	if declaredLen > 0 {
		// Use declared length minus current offset and any fields after this array
		// For now, assume no fields after - this will be refined when we have full pattern context
		availableBytes = declaredLen - offset
	} else if maxLen > 0 {
		availableBytes = maxLen - offset
	} else {
		// No length constraint - read all available data
		availableBytes = len(data) - offset
	}

	if availableBytes < 0 {
		availableBytes = 0
	}

	// Read as many complete fields as fit
	curr := offset
	readCount := 0
	maxFields := availableBytes / fieldSize

	for readCount < maxFields && curr+fieldSize <= len(data) {
		value, err := decodeField(data[curr:curr+fieldSize], n.spec)
		if err != nil {
			break
		}
		var name string
		if n.baseName != "" {
			name = fmt.Sprintf("%s_%d", n.baseName, readCount)
		} else {
			name = fmt.Sprintf("%s_%d", n.spec.Name, readCount)
		}
		state.AddField(DecodedField{
			Name:   name,
			Offset: curr,
			Value:  value,
			Type:   n.spec.Type,
		})
		curr += fieldSize
		readCount++
	}

	if readCount == 0 && curr+fieldSize > len(data) {
		return statusNeedMore, 0, nil
	}

	return statusMatch, curr, nil
}

type arrayStrideNode struct {
	spec     FieldSpec
	baseName string
	count    int
	stride   int
}

func (n *arrayStrideNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	elemSize := n.spec.TotalBytes()
	curr := offset
	for i := 0; i < n.count; i++ {
		if curr+elemSize > len(data) {
			return statusNeedMore, 0, nil
		}
		value, err := decodeField(data[curr:curr+elemSize], n.spec)
		if err != nil {
			return statusMatch, 0, err
		}
		name := fmt.Sprintf("%s_%d", n.spec.Name, i)
		if n.baseName != "" {
			name = fmt.Sprintf("%s_%d", n.baseName, i)
		}
		state.AddField(DecodedField{
			Name:   name,
			Offset: curr,
			Value:  value,
			Type:   n.spec.Type,
		})
		curr += n.stride
	}
	return statusMatch, curr, nil
}

type arrayOfStructsNode struct {
	count       int
	useStar     bool
	baseName    string
	structSpecs []FieldSpec
}

func (n *arrayOfStructsNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	// Calculate struct size
	structSize := 0
	for _, spec := range n.structSpecs {
		size := spec.TotalBytes()
		if size == 0 {
			return statusMatch, 0, fmt.Errorf("array of structs with variable-length fields not yet supported")
		}
		structSize += size
	}

	if structSize == 0 {
		return statusMatch, 0, fmt.Errorf("empty struct in array")
	}

	curr := offset
	count := n.count

	if n.useStar {
		// Read until exhausted - calculate remaining bytes
		// This is complex - for now, read until we can't read a complete struct
		index := 0
		for curr+structSize <= len(data) {
			// Read one struct
			structOffset := curr
			for _, spec := range n.structSpecs {
				size := spec.TotalBytes()
				value, err := decodeField(data[structOffset:structOffset+size], spec)
				if err != nil {
					// Can't read more structs
					if curr == offset {
						return statusNeedMore, 0, nil
					}
					return statusMatch, curr, nil
				}
				fieldName := spec.Name
				if n.baseName != "" {
					fieldName = fmt.Sprintf("%s_%d_%s", n.baseName, index, spec.Name)
				}
				state.AddField(DecodedField{
					Name:   fieldName,
					Offset: structOffset,
					Value:  value,
					Type:   spec.Type,
				})
				structOffset += size
			}
			curr += structSize
			index++
		}

		if curr == offset {
			return statusNeedMore, 0, nil
		}
		return statusMatch, curr, nil
	}

	// Fixed count array
	if curr+structSize*count > len(data) {
		return statusNeedMore, 0, nil
	}

	for i := 0; i < count; i++ {
		structOffset := curr
		for _, spec := range n.structSpecs {
			size := spec.TotalBytes()
			value, err := decodeField(data[structOffset:structOffset+size], spec)
			if err != nil {
				return statusMatch, 0, err
			}
			fieldName := fmt.Sprintf("%s_%d", spec.Name, i)
			if n.baseName != "" {
				fieldName = fmt.Sprintf("%s_%d_%s", n.baseName, i, spec.Name)
			}
			state.AddField(DecodedField{
				Name:   fieldName,
				Offset: structOffset,
				Value:  value,
				Type:   spec.Type,
			})
			structOffset += size
		}
		curr += structSize
	}

	return statusMatch, curr, nil
}

type expressionNode struct {
	name        string
	expr        expression
	exprSrc     string
	baseSpec    *FieldSpec
	baseType    string
	isCondition bool
}

func (n *expressionNode) eval(ctx *evalContext, data []byte, offset int, state State) (evalStatus, int, error) {
	curr := offset
	var baseValue exprValue
	var rawValue interface{}
	if n.baseSpec != nil {
		size := n.baseSpec.TotalBytes()
		if curr+size > len(data) {
			return statusNeedMore, 0, nil
		}
		value, err := decodeField(data[curr:curr+size], *n.baseSpec)
		if err != nil {
			return statusMatch, 0, err
		}
		rawValue = value
		curr += size
		val, ok := toExprValue(value)
		if !ok {
			return statusMatch, 0, fmt.Errorf("field %s cannot be used in expressions", n.baseSpec.Name)
		}
		baseValue = val
	}
	env := &exprEnv{
		state:        state,
		currentName:  n.name,
		currentValue: baseValue,
		baseType:     n.baseType,
	}
	result, err := n.expr.eval(env)
	if err != nil {
		return statusMatch, 0, err
	}
	if result.isBool {
		if !result.bool {
			return statusMatch, 0, fmt.Errorf("expression %s evaluated to false", n.exprSrc)
		}
		if !n.isCondition && n.baseSpec != nil && n.name != "" {
			state.AddField(DecodedField{
				Name:   n.name,
				Offset: offset,
				Value:  rawValue,
				Type:   n.baseSpec.Type,
			})
		}
		return statusMatch, curr, nil
	}
	fieldName := n.name
	if fieldName == "" {
		fieldName = fmt.Sprintf("expr_%d", len(state.Fields())+1)
	}
	state.AddField(DecodedField{
		Name:   fieldName,
		Offset: offset,
		Value:  result.num,
		Type:   FieldFloat64,
	})
	return statusMatch, curr, nil
}

type evalContext struct{}

// decodeVarintU and decodeVarintS are defined in ops.go
