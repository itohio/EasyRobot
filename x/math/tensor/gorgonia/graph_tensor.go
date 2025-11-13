package gorgonia

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// GraphTensor implements types.Tensor for Gorgonia graph execution.
// Operations on GraphTensor record operations in the graph rather than executing immediately.
type GraphTensor struct {
	graph    *ExpressionGraph
	node     *gorgonia.Node
	id       int
	shape    types.Shape
	dataType types.DataType
	constant bool
}

// Verify GraphTensor implements types.Tensor at compile time
var _ types.Tensor = (*GraphTensor)(nil)

// Graph returns the parent ExpressionGraph.
func (gt *GraphTensor) Graph() types.ExpressionGraph {
	return gt.graph
}

// IsConstant returns true if this is a constant tensor.
func (gt *GraphTensor) IsConstant() bool {
	return gt.constant
}

//
// Core Interface Implementation
//

// ID returns the tensor's unique identifier.
func (gt *GraphTensor) ID() uintptr {
	return uintptr(gt.id)
}

// DataType returns the tensor's data type.
func (gt *GraphTensor) DataType() types.DataType {
	return gt.dataType
}

// Data returns the tensor's data (only available after compilation).
func (gt *GraphTensor) Data() any {
	if gt.graph.State() != types.GraphCompiled && gt.graph.State() != types.GraphExecuting {
		panic("gorgonia.GraphTensor.Data: can only access data after compilation")
	}

	val := gt.node.Value()
	if val == nil {
		panic("gorgonia.GraphTensor.Data: node has no value")
	}

	// Get underlying tensor
	dense, ok := val.(*tensor.Dense)
	if !ok {
		panic("gorgonia.GraphTensor.Data: value is not a Dense tensor")
	}

	return dense.Data()
}

// Shape returns a copy of the tensor's shape.
func (gt *GraphTensor) Shape() types.Shape {
	return gt.shape.Clone()
}

// Rank returns the number of dimensions.
func (gt *GraphTensor) Rank() int {
	return len(gt.shape)
}

// Size returns the total number of elements.
func (gt *GraphTensor) Size() int {
	return gt.shape.Size()
}

// Empty returns true if the tensor is empty.
func (gt *GraphTensor) Empty() bool {
	return len(gt.shape) == 0 || gt.shape.Size() == 0
}

// Strides returns the tensor's strides.
func (gt *GraphTensor) Strides(dst []int) []int {
	return gt.shape.Strides(dst)
}

// IsContiguous returns true for graph tensors (managed by Gorgonia).
func (gt *GraphTensor) IsContiguous() bool {
	return true
}

// Offset returns 0 for graph tensors.
func (gt *GraphTensor) Offset() int {
	return 0
}

// DataWithOffset returns the data (same as Data() for graph tensors).
func (gt *GraphTensor) DataWithOffset() any {
	return gt.Data()
}

// At returns element at given indices (only after compilation).
func (gt *GraphTensor) At(indices ...int) float64 {
	if gt.graph.State() != types.GraphCompiled {
		panic("gorgonia.GraphTensor.At: can only access data after compilation and execution")
	}

	val := gt.node.Value()
	if val == nil {
		panic("gorgonia.GraphTensor.At: node has no value")
	}

	dense, ok := val.(*tensor.Dense)
	if !ok {
		panic("gorgonia.GraphTensor.At: value is not a Dense tensor")
	}

	// Handle linear indexing
	if len(indices) == 1 && gt.Rank() > 1 {
		linearIdx := indices[0]
		indices = make([]int, gt.Rank())
		remaining := linearIdx
		for i := gt.Rank() - 1; i >= 0; i-- {
			indices[i] = remaining % gt.shape[i]
			remaining /= gt.shape[i]
		}
	}

	val2, err := dense.At(indices...)
	if err != nil {
		panic(err)
	}

	// Convert to float64
	switch v := val2.(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	default:
		panic(fmt.Sprintf("unsupported type: %T", v))
	}
}

// SetAt sets element at given indices (only for non-constant tensors before execution).
func (gt *GraphTensor) SetAt(value float64, indices ...int) {
	if gt.constant {
		panic("gorgonia.GraphTensor.SetAt: cannot modify constant tensor")
	}

	if gt.graph.State() != types.GraphCompiled {
		panic("gorgonia.GraphTensor.SetAt: can only set data after compilation, before execution")
	}

	val := gt.node.Value()
	if val == nil {
		panic("gorgonia.GraphTensor.SetAt: node has no value")
	}

	dense, ok := val.(*tensor.Dense)
	if !ok {
		panic("gorgonia.GraphTensor.SetAt: value is not a Dense tensor")
	}

	// Handle linear indexing
	if len(indices) == 1 && gt.Rank() > 1 {
		linearIdx := indices[0]
		indices = make([]int, gt.Rank())
		remaining := linearIdx
		for i := gt.Rank() - 1; i >= 0; i-- {
			indices[i] = remaining % gt.shape[i]
			remaining /= gt.shape[i]
		}
	}

	// Convert value to appropriate type
	var val2 any
	switch gt.dataType {
	case types.FP32:
		val2 = float32(value)
	case types.FP64:
		val2 = value
	case types.INT, types.INT32:
		val2 = int32(value)
	case types.INT64:
		val2 = int64(value)
	default:
		val2 = float32(value)
	}

	if err := dense.SetAt(val2, indices...); err != nil {
		panic(err)
	}
}

// Elements creates an iterator over tensor elements.
func (gt *GraphTensor) Elements(fixedAxisValuePairs ...int) func(func(types.Element) bool) {
	// For graph tensors, elements can only be accessed after execution
	return func(yield func(types.Element) bool) {
		if gt.graph.State() != types.GraphCompiled {
			panic("gorgonia.GraphTensor.Elements: can only iterate after compilation and execution")
		}

		val := gt.node.Value()
		if val == nil {
			return
		}

		dense, ok := val.(*tensor.Dense)
		if !ok {
			return
		}

		// Simple iteration over all elements
		it := dense.Iterator()
		for _, err := it.Next(); err == nil; _, err = it.Next() {
			coord := it.Coord()
			elem := &graphElement{tensor: gt, indices: coord}
			if !yield(elem) {
				return
			}
		}
	}
}

// Release is a no-op for graph tensors (Gorgonia manages memory).
func (gt *GraphTensor) Release() {
	// No-op: Gorgonia manages memory
}

// graphElement implements types.Element for graph tensors
type graphElement struct {
	tensor  *GraphTensor
	indices []int
}

func (e *graphElement) Get() float64 {
	return e.tensor.At(e.indices...)
}

func (e *graphElement) Set(value float64) {
	e.tensor.SetAt(value, e.indices...)
}
