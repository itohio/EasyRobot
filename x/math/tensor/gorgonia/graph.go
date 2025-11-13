package gorgonia

import (
	"fmt"
	"sync"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ExpressionGraph implements types.ExecutionGraph for Gorgonia.
// It wraps gorgonia.ExprGraph and provides tensor-based operations.
//
// Usage:
//
//	eg := gorgonia.NewExpressionGraph()
//	t1 := eg.New(types.FP32, 10, 10)
//	t2 := eg.New(types.FP32, 10, 10)
//	result := t1.MatMul(nil, t2).ReLU(nil)
//	eg.Compile()
//	t1.Copy(data1)
//	t2.Copy(data2)
//	eg.Compute()
//	output := result.Data()
type ExpressionGraph struct {
	mu sync.RWMutex

	graph   *gorgonia.ExprGraph
	vm      gorgonia.VM
	state   types.GraphState
	tensors []*GraphTensor
	nextID  int
}

// NewExpressionGraph creates a new Gorgonia expression graph.
func NewExpressionGraph() *ExpressionGraph {
	return &ExpressionGraph{
		graph:   gorgonia.NewGraph(),
		state:   types.GraphBuilding,
		tensors: make([]*GraphTensor, 0),
		nextID:  0,
	}
}

// New creates a new tensor node in the graph.
func (eg *ExpressionGraph) New(dtype types.DataType, shape ...int) types.Tensor {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	if eg.state != types.GraphBuilding {
		panic("gorgonia.ExpressionGraph.New: cannot add tensors after compilation")
	}

	// Convert to Gorgonia dtype
	gdt := toGorgoniaDtype(dtype)

	// Create node in the graph
	gshape := make([]int, len(shape))
	copy(gshape, shape)

	node := gorgonia.NewTensor(
		eg.graph,
		gdt,
		len(shape),
		gorgonia.WithShape(gshape...),
		gorgonia.WithName(fmt.Sprintf("tensor_%d", eg.nextID)),
	)

	gt := &GraphTensor{
		graph:    eg,
		node:     node,
		id:       eg.nextID,
		shape:    types.Shape(shape),
		dataType: dtype,
		constant: false,
	}

	eg.tensors = append(eg.tensors, gt)
	eg.nextID++

	return gt
}

// NewConstant creates a constant tensor node with embedded data.
func (eg *ExpressionGraph) NewConstant(data any, shape ...int) types.Tensor {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	if eg.state != types.GraphBuilding {
		panic("gorgonia.ExpressionGraph.NewConstant: cannot add tensors after compilation")
	}

	// Infer dtype from data
	dtype := types.TypeFromData(data)
	gdt := toGorgoniaDtype(dtype)

	// Create constant node
	gshape := make([]int, len(shape))
	copy(gshape, shape)

	// Create dense tensor with data
	denseVal := tensor.New(tensor.WithShape(gshape...), tensor.Of(gdt), tensor.WithBacking(data))

	node := gorgonia.NewTensor(
		eg.graph,
		gdt,
		len(shape),
		gorgonia.WithShape(gshape...),
		gorgonia.WithValue(denseVal),
		gorgonia.WithName(fmt.Sprintf("const_%d", eg.nextID)),
	)

	gt := &GraphTensor{
		graph:    eg,
		node:     node,
		id:       eg.nextID,
		shape:    types.Shape(shape),
		dataType: dtype,
		constant: true,
	}

	eg.tensors = append(eg.tensors, gt)
	eg.nextID++

	return gt
}

// State returns the current state of the graph.
func (eg *ExpressionGraph) State() types.GraphState {
	eg.mu.RLock()
	defer eg.mu.RUnlock()
	return eg.state
}

// Compile compiles the graph for execution.
func (eg *ExpressionGraph) Compile() error {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	if eg.state != types.GraphBuilding {
		return fmt.Errorf("graph already compiled or executing")
	}

	// Create tape machine for execution
	eg.vm = gorgonia.NewTapeMachine(eg.graph)
	eg.state = types.GraphCompiled

	return nil
}

// Compute executes the compiled graph.
func (eg *ExpressionGraph) Compute() error {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	if eg.state != types.GraphCompiled {
		return fmt.Errorf("graph must be compiled before execution")
	}

	eg.state = types.GraphExecuting
	defer func() { eg.state = types.GraphCompiled }()

	// Run the VM
	if err := eg.vm.RunAll(); err != nil {
		return fmt.Errorf("execution failed: %w", err)
	}

	// Reset for next execution
	eg.vm.Reset()

	return nil
}

// Reset resets the graph to building state.
func (eg *ExpressionGraph) Reset() {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	if eg.vm != nil {
		eg.vm.Close()
		eg.vm = nil
	}

	eg.graph = gorgonia.NewGraph()
	eg.state = types.GraphBuilding
	eg.tensors = make([]*GraphTensor, 0)
	eg.nextID = 0
}

// TensorCount returns the number of tensors in the graph.
func (eg *ExpressionGraph) TensorCount() int {
	eg.mu.RLock()
	defer eg.mu.RUnlock()
	return len(eg.tensors)
}

// OperationCount returns the number of operations in the graph.
func (eg *ExpressionGraph) OperationCount() int {
	eg.mu.RLock()
	defer eg.mu.RUnlock()

	// Count nodes in the graph (approximation)
	return eg.graph.AllNodes().Len()
}

// Helper to convert types.DataType to tensor dtype
func toGorgoniaDtype(dt types.DataType) tensor.Dtype {
	switch dt {
	case types.FP32:
		return tensor.Float32
	case types.FP64:
		return tensor.Float64
	case types.INT:
		return tensor.Int
	case types.INT32:
		return tensor.Int32
	case types.INT64:
		return tensor.Int64
	default:
		return tensor.Float32 // Default fallback
	}
}

// getTensor finds a GraphTensor by ID
func (eg *ExpressionGraph) getTensor(id int) *GraphTensor {
	for _, t := range eg.tensors {
		if t.id == id {
			return t
		}
	}
	return nil
}
