# Gorgonia Tensor Wrapper Architecture

## The Problem

Gorgonia has **two distinct APIs**:

### 1. Tensor API (`gorgonia.org/tensor`)
- Direct tensor operations: `tensor.Add()`, `tensor.MatMul()`, etc.
- **Does NOT have**: Conv2D, MaxPool2D, complex CNN operations
- Used for: Basic math on tensors

### 2. Graph API (`gorgonia.org/gorgonia`)  
- Computation graph: `gorgonia.NewGraph()`, nodes, compile, execute
- **Has**: Conv2D, MaxPool2D, all CNN operations
- Used for: Building and executing neural networks

## The Solution: Graph-Based Wrapper

Gorgonia is **fundamentally a graph-based library**. We should wrap the graph API, not the tensor API.

### Pattern

```go
// Create computation graph
eg := gorgonia.NewExpressionGraph()

// Create tensor nodes (placeholders)
input := eg.New(types.FP32, 1, 28, 28)  // MNIST image
weights1 := eg.NewConstant(w1Data, 32, 1, 3, 3)
bias1 := eg.NewConstant(b1Data, 32)

// Build computation graph (records ops, doesn't execute)
conv1 := input.Conv2D(nil, weights1, bias1, []int{1,1}, []int{1,1})
relu1 := conv1.ReLU(nil)
pool1 := relu1.MaxPool2D(nil, []int{2,2}, []int{2,2}, []int{0,0})

// Compile graph once
eg.Compile()

// Execute multiple times with different inputs
for batch := range batches {
    input.Copy(batch.Images)  // Set input data
    eg.Compute()              // Execute compiled graph
    output := pool1.Data()    // Get output data
}
```

### Key Benefits

1. **Native Gorgonia**: Uses Gorgonia's graph API directly
2. **All Operations Available**: Conv2D, pooling, everything Gorgonia supports
3. **Efficient**: Compile once, execute many times
4. **Optimized**: Gorgonia can optimize the computation graph
5. **Memory Efficient**: Gorgonia manages memory internally

## Implementation Plan

### 1. Create `tensor/types/graph.go` ✅
Define the graph execution interface that backends implement:
- `ExecutionGraph` interface
- `GraphTensor` interface  
- `GraphBackend` interface

### 2. Implement `gorgonia/graph.go`

```go
type ExpressionGraph struct {
    graph   *gorgonia.ExprGraph
    vm      gorgonia.VM
    tensors map[int]*GraphTensor
    state   GraphState
}

type GraphTensor struct {
    graph    *ExpressionGraph
    node     *gorgonia.Node
    id       int
    shape    types.Shape
    dataType types.DataType
}
```

### 3. Wire Operations

Each tensor operation creates a Gorgonia node:

```go
func (t GraphTensor) MatMul(dst types.Tensor, other types.Tensor) types.Tensor {
    if t.graph.State() != GraphBuilding {
        panic("cannot add operations after compilation")
    }
    
    otherGT := other.(GraphTensor)
    
    // Create Gorgonia node
    result := gorgonia.Must(gorgonia.Mul(t.node, otherGT.node))
    
    // Wrap in GraphTensor
    return t.graph.wrapNode(result)
}
```

### 4. Compilation

```go
func (eg *ExpressionGraph) Compile() error {
    // Create tape machine for execution
    eg.vm = gorgonia.NewTapeMachine(eg.graph)
    eg.state = GraphCompiled
    return nil
}
```

### 5. Execution

```go
func (eg *ExpressionGraph) Compute() error {
    // Run the computation
    return eg.vm.RunAll()
}
```

## Current State

### What We Have ✅
- Basic tensor operations (Add, Mul, MatMul) via `tensor` package
- Activations (ReLU, Sigmoid, Tanh, LeakyReLU, GELU, etc.) - manual implementation
- Benchmarks showing Gorgonia is 2-43x faster for MatMul

### What We Need 🚧
- Graph-based wrapper (`gorgonia/graph.go`)
- Implementation of `ExecutionGraph` interface
- Wire all operations to Gorgonia graph nodes
- Handle constant vs variable tensors
- Memory management for graph tensors

### Operations Available in Gorgonia Graph API
- ✅ Conv2D
- ✅ MaxPool2D
- ✅ BatchNorm
- ✅ Dropout
- ✅ All activations (via graph nodes)
- ✅ All math operations
- ✅ Automatic differentiation (for training)

## Migration Path

### Phase 1: Graph Wrapper (Current Priority)
1. Implement `types/graph.go` interface ✅
2. Implement `gorgonia/graph.go`
3. Test with simple operations (Add, MatMul)
4. Test with CNN operations (Conv2D, Pool)

### Phase 2: Integration
1. Update layer implementations to work with both eager and graph modes
2. Add graph compilation to model loading
3. Benchmark graph vs eager execution

### Phase 3: Optimization
1. Graph optimizations (fusion, memory reuse)
2. Multi-threaded execution
3. GPU support (via Gorgonia's CUDA backend)

## Example: Full CNN

```go
// Load model weights
loader := keras.NewLoader()
model, _ := loader.LoadFromFile("resnet50.h5")

// Create Gorgonia execution graph
eg := gorgonia.NewExpressionGraph()

// Build computation graph from model
input := eg.New(types.FP32, 1, 3, 224, 224)
output := model.BuildGraph(eg, input)  // Model creates graph

// Compile once
eg.Compile()

// Run inference on many images
for img := range images {
    input.Copy(img)
    eg.Compute()
    predictions := output.Data().([]float32)
    processResults(predictions)
}
```

## Comparison with Current Approach

### Current (Eager-style Tensor Wrapper)
- ❌ Missing CNN operations (Conv2D, pooling)
- ❌ Not using Gorgonia's strengths
- ✅ Simple mental model
- ✅ Easy debugging

### Proposed (Graph-based Wrapper)
- ✅ All operations available
- ✅ Native Gorgonia design
- ✅ Compile once, run many (fast)
- ✅ Graph optimizations
- ⚠️  More complex mental model
- ⚠️  Harder debugging (graph vs eager)

## Recommendation

**Implement the graph-based wrapper.** This aligns with:
1. Gorgonia's design philosophy
2. TFLite's existing graph implementation
3. Modern ML framework patterns (TensorFlow, PyTorch JIT)
4. Performance requirements for inference

The eager tensor wrapper should be kept for:
- Quick prototyping
- Debugging
- Simple operations
- Python-like usage

But production inference should use the graph-based wrapper for performance and access to all operations.

