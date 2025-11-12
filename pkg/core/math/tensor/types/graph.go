package types

// GraphState represents the state of a computation graph.
type GraphState int

const (
	// GraphBuilding state: recording operations into the graph
	GraphBuilding GraphState = iota
	// GraphCompiled state: graph is compiled and ready to execute
	GraphCompiled
	// GraphExecuting state: graph is currently executing
	GraphExecuting
)

// ExpressionGraph represents a computation graph that records operations
// and compiles them for efficient execution.
//
// Pattern:
//
//	eg := backend.NewExpressionGraph()
//	t1 := eg.New(types.FP32, 10, 10)
//	t2 := eg.New(types.FP32, 10, 10)
//	result := t1.MatMul(nil, t2).ReLU(nil)  // Records ops, doesn't execute
//	eg.Compile()                             // Compile once
//	for i := 0; i i < 100; i++ {
//	    t1.Copy(inputData1[i])               // Set inputs
//	    t2.Copy(inputData2[i])
//	    eg.Compute()                         // Execute compiled graph
//	    output := result.Data()              // Get outputs
//	}
type ExpressionGraph interface {
	// New creates a new tensor node in the graph.
	// The tensor is initially empty but can receive data via Copy/SetAt.
	New(dtype DataType, shape ...int) Tensor

	// NewConstant creates a constant tensor node in the graph.
	// The data is embedded in the compiled graph and cannot be changed.
	NewConstant(data any, shape ...int) Tensor

	// State returns the current state of the graph.
	State() GraphState

	// Compile compiles the recorded operations into an optimized execution plan.
	// After compilation, no new operations can be added to the graph.
	// Must be called before Compute().
	Compile() error

	// Compute executes the compiled graph with the current input data.
	// Must be called after Compile().
	// Can be called multiple times with different input data (set via Copy/SetAt).
	Compute() error

	// Reset resets the graph to Building state, clearing all operations.
	// Use this to rebuild a new graph.
	Reset()

	// TensorCount returns the number of tensor nodes in the graph.
	TensorCount() int

	// OperationCount returns the number of operations recorded in the graph.
	OperationCount() int
}

// GraphTensor is a tensor that belongs to an ExpressionGraph.
// Operations on GraphTensors are recorded in the graph rather than executed immediately.
//
// A GraphTensor implements the Tensor interface, but:
// - Operations (Add, MatMul, etc.) record operations in the graph and return new GraphTensors
// - Data access (Data, At, SetAt) only works after the graph is compiled
// - Copy() can set input data before Compute()
type GraphTensor interface {
	Tensor

	// Graph returns the parent ExpressionGraph.
	Graph() ExpressionGraph

	// IsConstant returns true if this tensor is a constant (data embedded in graph).
	IsConstant() bool
}

// Backend capabilities for determining which execution mode to use.
type BackendCapabilities struct {
	// SupportsEagerExecution indicates the backend can execute operations immediately.
	SupportsEagerExecution bool

	// SupportsGraphExecution indicates the backend can record and compile computation graphs.
	SupportsGraphExecution bool

	// RecommendedMode indicates which mode is recommended for this backend.
	// For Gorgonia: GraphExecution (it's designed for graphs)
	// For eager_tensor: EagerExecution (it's designed for eager)
	// For TFLite: GraphExecution (must compile to flatbuffer)
	RecommendedMode ExecutionMode
}

type ExecutionMode int

const (
	EagerMode ExecutionMode = iota
	GraphMode
)

// GraphBackend is an optional interface that tensor implementations can provide
// to expose graph-based execution capabilities.
type GraphBackend interface {
	// NewExpressionGraph creates a new computation graph for this backend.
	NewExpressionGraph() ExpressionGraph

	// Capabilities returns information about what this backend supports.
	Capabilities() BackendCapabilities
}
