package peg

import (
	"github.com/itohio/EasyRobot/x/math/graph"
)

type nodeData struct {
	Description string
}

// Program represents a compiled PEG pattern backed by an expression tree.
type Program struct {
	*graph.GenericExpressionGraph[nodeData, float32, State, Decision]
	root        node
	anchorStart bool
	anchorEnd   bool
	maxLen      int
	rawPattern  string
}

func newProgram(pattern string, root node, anchorStart, anchorEnd bool, maxLen int) (*Program, error) {
	exprTree := graph.NewGenericExpressionGraph[nodeData, float32, State, Decision]()

	op := func(state State, _ map[int64]Decision) (Decision, bool) {
		if state == nil {
			state = NewDefaultState()
		}
		ctx := &evalContext{}
		packet := state.Packet()
		status, end, err := root.eval(ctx, packet, 0, state)
		if err != nil {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true
		}

		if status == statusNeedMore {
			if maxLen > 0 && len(packet) > maxLen {
				state.SetDecision(DecisionDrop)
				return DecisionDrop, true
			}
			if state.MaxLength() > 0 && len(packet) > state.MaxLength() {
				state.SetDecision(DecisionDrop)
				return DecisionDrop, true
			}
			state.SetDecision(DecisionContinue)
			return DecisionContinue, true
		}

		if maxLen > 0 && len(packet) > maxLen {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true
		}
		if state.MaxLength() > 0 && len(packet) > state.MaxLength() {
			state.SetDecision(DecisionDrop)
			return DecisionDrop, true
		}

		declared := state.DeclaredLength()
		if declared > 0 {
			if len(packet) < declared {
				state.SetDecision(DecisionContinue)
				return DecisionContinue, true
			}
			if len(packet) > declared {
				state.SetDecision(DecisionDrop)
				return DecisionDrop, true
			}
		}

		// For end anchor, check that we've consumed all bytes
		// When we jump backward, the returned 'end' might be less than packet length
		// but we still need to ensure all bytes were part of the match
		// For patterns with backward jumps, we check if all bytes were read through fields
		if anchorEnd {
			finalOffset := state.Offset()
			// If final offset or returned end is at packet length, we're good
			if finalOffset == len(packet) || end == len(packet) {
				// Normal case - consumed all bytes sequentially
			} else {
				// Might be a backward jump case - check if all bytes were read through fields
				fields := state.Fields()
				maxFieldEnd := 0
				for _, f := range fields {
					// Calculate field end offset
					fieldEnd := f.Offset
					switch f.Type {
					case FieldUint8, FieldInt8:
						fieldEnd += 1
					case FieldUint16LE, FieldUint16BE, FieldInt16LE, FieldInt16BE:
						fieldEnd += 2
					case FieldUint32LE, FieldUint32BE, FieldInt32LE, FieldInt32BE, FieldFloat32:
						fieldEnd += 4
					case FieldUint64LE, FieldUint64BE, FieldInt64LE, FieldInt64BE, FieldFloat64:
						fieldEnd += 8
					case FieldStringFixed:
						if str, ok := f.Value.(string); ok {
							fieldEnd += len(str)
						}
					case FieldStringPascal:
						if str, ok := f.Value.(string); ok {
							fieldEnd += 1 + len(str) // length byte + string
						}
					case FieldStringC:
						if str, ok := f.Value.(string); ok {
							fieldEnd += len(str) + 1 // string + null terminator
						}
					default:
						// For other types, try to estimate size
						fieldEnd += 1 // at least 1 byte
					}
					if fieldEnd > maxFieldEnd {
						maxFieldEnd = fieldEnd
					}
				}
				// Also check literals and other consumed bytes
				// For now, if maxFieldEnd doesn't cover the packet and neither offset is at end, drop it
				if maxFieldEnd < len(packet) {
					state.SetDecision(DecisionDrop)
					return DecisionDrop, true
				}
			}
		}

		state.SetDecision(DecisionEmit)
		return DecisionEmit, true
	}

	rootNode, err := exprTree.AddNode(nodeData{Description: "peg_root"}, op)
	if err != nil {
		return nil, err
	}
	exprTree.SetRoot(rootNode)

	return &Program{
		GenericExpressionGraph: exprTree,
		root:                   root,
		anchorStart:            anchorStart,
		anchorEnd:              anchorEnd,
		maxLen:                 maxLen,
		rawPattern:             pattern,
	}, nil
}

// Decide evaluates the pattern for the current packet stored in state.
func (p *Program) Decide(state State) (Decision, error) {
	results, err := p.Compute(nil, state)
	if err != nil {
		return DecisionDrop, err
	}
	if len(results) == 0 {
		return DecisionDrop, nil
	}
	return results[0], nil
}
