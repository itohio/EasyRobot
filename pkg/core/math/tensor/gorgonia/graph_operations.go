package gorgonia

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// This file implements tensor operations for GraphTensor using Gorgonia's graph API.
// All operations record nodes in the computation graph rather than executing immediately.

//
// Manipulation Operations
//

// Clone creates a new node that copies this tensor.
func (gt *GraphTensor) Clone() types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Clone: can only add operations during graph building")
	}

	// Create identity operation by adding zero (node + 0)
	zero := gorgonia.NewConstant(0.0, gorgonia.WithName("zero_clone"))
	node, err := gorgonia.Add(gt.node, zero)
	if err != nil {
		panic(err)
	}

	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Copy copies data from src (can be used to set input data after compilation).
func (gt *GraphTensor) Copy(src types.Tensor) types.Tensor {
	if gt.constant {
		panic("gorgonia.GraphTensor.Copy: cannot modify constant tensor")
	}

	// If we're in compiled state, this is setting input data
	if gt.graph.State() == types.GraphCompiled {
		// Get the source data
		srcData := src.Data()

		// Set our node's value
		val := gt.node.Value()
		if val == nil {
			// Initialize the tensor
			gdt := toGorgoniaDtype(gt.dataType)
			denseVal := tensor.New(tensor.WithShape(gt.shape...), tensor.Of(gdt), tensor.WithBacking(srcData))
			gorgonia.Let(gt.node, denseVal)
		} else {
			// Copy data into existing tensor
			dense := val.(*tensor.Dense)
			switch gt.dataType {
			case types.FP32:
				dstData := dense.Data().([]float32)
				srcSlice := srcData.([]float32)
				copy(dstData, srcSlice)
			case types.FP64:
				dstData := dense.Data().([]float64)
				srcSlice := srcData.([]float64)
				copy(dstData, srcSlice)
			default:
				panic(fmt.Sprintf("unsupported data type for copy: %v", gt.dataType))
			}
		}
		return gt
	}

	// During graph building, create a copy operation
	if gt.graph.State() == types.GraphBuilding {
		// This is a graph operation - not implemented yet
		panic("gorgonia.GraphTensor.Copy: graph-time copy not yet implemented")
	}

	panic("gorgonia.GraphTensor.Copy: invalid state")
}

// Fill fills the tensor with a constant value (during graph building).
func (gt *GraphTensor) Fill(dst types.Tensor, value float64) types.Tensor {
	// TODO: Implement as graph operation
	panic("gorgonia.GraphTensor.Fill: not yet implemented for graph tensors")
}

// Reshape reshapes the tensor.
func (gt *GraphTensor) Reshape(dst types.Tensor, newShape types.Shape) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Reshape: can only add operations during graph building")
	}

	if gt.Size() != newShape.Size() {
		panic("gorgonia.GraphTensor.Reshape: size mismatch")
	}

	node, err := gorgonia.Reshape(gt.node, tensor.Shape(newShape))
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, newShape, gt.dataType)
}

// Transpose transposes the tensor.
func (gt *GraphTensor) Transpose(dst types.Tensor, dims []int) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Transpose: can only add operations during graph building")
	}

	if dims == nil {
		// Default: reverse all dimensions
		dims = make([]int, gt.Rank())
		for i := range dims {
			dims[i] = gt.Rank() - 1 - i
		}
	}

	node, err := gorgonia.Transpose(gt.node, dims...)
	if err != nil {
		panic(err)
	}

	// Compute new shape
	newShape := make(types.Shape, len(gt.shape))
	for i, d := range dims {
		newShape[i] = gt.shape[d]
	}

	return gt.graph.wrapNode(node, newShape, gt.dataType)
}

//
// Element-wise Binary Operations
//

// Add performs element-wise addition.
func (gt *GraphTensor) Add(dst types.Tensor, other types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Add: can only add operations during graph building")
	}

	otherGT, ok := other.(*GraphTensor)
	if !ok {
		panic("gorgonia.GraphTensor.Add: other must be a GraphTensor")
	}

	node, err := gorgonia.Add(gt.node, otherGT.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Subtract performs element-wise subtraction.
func (gt *GraphTensor) Subtract(dst types.Tensor, other types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Subtract: can only add operations during graph building")
	}

	otherGT, ok := other.(*GraphTensor)
	if !ok {
		panic("gorgonia.GraphTensor.Subtract: other must be a GraphTensor")
	}

	node, err := gorgonia.Sub(gt.node, otherGT.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Multiply performs element-wise multiplication.
func (gt *GraphTensor) Multiply(dst types.Tensor, other types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Multiply: can only add operations during graph building")
	}

	otherGT, ok := other.(*GraphTensor)
	if !ok {
		panic("gorgonia.GraphTensor.Multiply: other must be a GraphTensor")
	}

	node, err := gorgonia.HadamardProd(gt.node, otherGT.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Divide performs element-wise division.
func (gt *GraphTensor) Divide(dst types.Tensor, other types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Divide: can only add operations during graph building")
	}

	otherGT, ok := other.(*GraphTensor)
	if !ok {
		panic("gorgonia.GraphTensor.Divide: other must be a GraphTensor")
	}

	node, err := gorgonia.HadamardDiv(gt.node, otherGT.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

//
// Scalar Operations
//

// ScalarMul multiplies by a scalar.
func (gt *GraphTensor) ScalarMul(dst types.Tensor, scalar float64) types.Tensor {
	return gt.MulScalar(dst, scalar)
}

// AddScalar adds a scalar to all elements.
func (gt *GraphTensor) AddScalar(dst types.Tensor, scalar float64) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.AddScalar: can only add operations during graph building")
	}

	var scalarNode *gorgonia.Node
	switch gt.dataType {
	case types.FP32:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	case types.FP64:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(scalar))
	default:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	}

	node, err := gorgonia.Add(gt.node, scalarNode)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// SubScalar subtracts a scalar from all elements.
func (gt *GraphTensor) SubScalar(dst types.Tensor, scalar float64) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.SubScalar: can only add operations during graph building")
	}

	var scalarNode *gorgonia.Node
	switch gt.dataType {
	case types.FP32:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	case types.FP64:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(scalar))
	default:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	}

	node, err := gorgonia.Sub(gt.node, scalarNode)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// MulScalar multiplies all elements by a scalar.
func (gt *GraphTensor) MulScalar(dst types.Tensor, scalar float64) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.MulScalar: can only add operations during graph building")
	}

	var scalarNode *gorgonia.Node
	switch gt.dataType {
	case types.FP32:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	case types.FP64:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(scalar))
	default:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	}

	node, err := gorgonia.HadamardProd(gt.node, scalarNode)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// DivScalar divides all elements by a scalar.
func (gt *GraphTensor) DivScalar(dst types.Tensor, scalar float64) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.DivScalar: can only add operations during graph building")
	}

	var scalarNode *gorgonia.Node
	switch gt.dataType {
	case types.FP32:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	case types.FP64:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(scalar))
	default:
		scalarNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(scalar)))
	}

	node, err := gorgonia.HadamardDiv(gt.node, scalarNode)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

//
// Unary Operations
//

// Square squares all elements.
func (gt *GraphTensor) Square(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Square: can only add operations during graph building")
	}

	node, err := gorgonia.Square(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Sqrt takes square root of all elements.
func (gt *GraphTensor) Sqrt(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Sqrt: can only add operations during graph building")
	}

	node, err := gorgonia.Sqrt(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Exp applies exponential to all elements.
func (gt *GraphTensor) Exp(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Exp: can only add operations during graph building")
	}

	node, err := gorgonia.Exp(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Log applies natural logarithm to all elements.
func (gt *GraphTensor) Log(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Log: can only add operations during graph building")
	}

	node, err := gorgonia.Log(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Pow raises all elements to a power.
func (gt *GraphTensor) Pow(dst types.Tensor, power float64) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Pow: can only add operations during graph building")
	}

	// Create scalar node for power
	var powerNode *gorgonia.Node
	switch gt.dataType {
	case types.FP32:
		powerNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(power)))
	case types.FP64:
		powerNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(power))
	default:
		powerNode = gorgonia.NewScalar(gt.graph.graph, toGorgoniaDtype(gt.dataType), gorgonia.WithValue(float32(power)))
	}

	node, err := gorgonia.Pow(gt.node, powerNode)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Abs applies absolute value to all elements.
func (gt *GraphTensor) Abs(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Abs: can only add operations during graph building")
	}

	node, err := gorgonia.Abs(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Negative negates all elements.
func (gt *GraphTensor) Negative(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Negative: can only add operations during graph building")
	}

	node, err := gorgonia.Neg(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

//
// Linear Algebra Operations
//

// MatMul performs matrix multiplication.
func (gt *GraphTensor) MatMul(dst types.Tensor, other types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.MatMul: can only add operations during graph building")
	}

	otherGT, ok := other.(*GraphTensor)
	if !ok {
		panic("gorgonia.GraphTensor.MatMul: other must be a GraphTensor")
	}

	node, err := gorgonia.Mul(gt.node, otherGT.node)
	if err != nil {
		panic(err)
	}

	// Compute output shape
	var outShape types.Shape
	if gt.Rank() == 2 && otherGT.Rank() == 2 {
		outShape = types.Shape{gt.shape[0], otherGT.shape[1]}
	} else {
		// TODO: Handle batched matmul
		panic("gorgonia.GraphTensor.MatMul: only 2D matmul supported currently")
	}

	return gt.graph.wrapNode(node, outShape, gt.dataType)
}

// Sum performs reduction sum.
func (gt *GraphTensor) Sum(dst types.Tensor, dims []int) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Sum: can only add operations during graph building")
	}

	var node *gorgonia.Node
	var err error
	if dims == nil || len(dims) == 0 {
		// Sum all elements
		node, err = gorgonia.Sum(gt.node)
		if err != nil {
			panic(err)
		}
		return gt.graph.wrapNode(node, types.Shape{}, gt.dataType)
	}

	// Sum along specific dimensions
	node, err = gorgonia.Sum(gt.node, dims...)
	if err != nil {
		panic(err)
	}

	// TODO: Compute output shape correctly
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

//
// Activation Functions
//

// ReLU applies ReLU activation.
func (gt *GraphTensor) ReLU(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.ReLU: can only add operations during graph building")
	}

	node, err := gorgonia.Rectify(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Sigmoid applies sigmoid activation.
func (gt *GraphTensor) Sigmoid(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Sigmoid: can only add operations during graph building")
	}

	node, err := gorgonia.Sigmoid(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Tanh applies tanh activation.
func (gt *GraphTensor) Tanh(dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Tanh: can only add operations during graph building")
	}

	node, err := gorgonia.Tanh(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// LeakyReLU applies LeakyReLU activation.
func (gt *GraphTensor) LeakyReLU(dst types.Tensor, alpha float64) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.LeakyReLU: can only add operations during graph building")
	}

	node, err := gorgonia.LeakyRelu(gt.node, alpha)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

// Softmax applies softmax activation.
func (gt *GraphTensor) Softmax(dim int, dst types.Tensor) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Softmax: can only add operations during graph building")
	}

	node, err := gorgonia.SoftMax(gt.node)
	if err != nil {
		panic(err)
	}
	return gt.graph.wrapNode(node, gt.shape, gt.dataType)
}

//
// Convolution Operations
//

// Conv2D performs 2D convolution.
func (gt *GraphTensor) Conv2D(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.Conv2D: can only add operations during graph building")
	}

	kernelGT, ok := kernel.(*GraphTensor)
	if !ok {
		panic("gorgonia.GraphTensor.Conv2D: kernel must be a GraphTensor")
	}

	// Conv2D in Gorgonia
	// Input: [batch, channels, height, width]
	// Kernel: [outChannels, inChannels, kernelH, kernelW]
	node, err := gorgonia.Conv2d(gt.node, kernelGT.node, tensor.Shape{stride[0], stride[1]}, []int{padding[0], padding[1]}, []int{1, 1}, []int{1, 1})
	if err != nil {
		panic(err)
	}

	// Add bias if provided
	if bias != nil {
		biasGT, ok := bias.(*GraphTensor)
		if !ok {
			panic("gorgonia.GraphTensor.Conv2D: bias must be a GraphTensor")
		}
		node, err = gorgonia.Add(node, biasGT.node)
		if err != nil {
			panic(err)
		}
	}

	// Compute output shape
	// TODO: Calculate proper output shape based on stride and padding
	outShape := gt.shape.Clone()

	return gt.graph.wrapNode(node, outShape, gt.dataType)
}

// MaxPool2D performs max pooling.
func (gt *GraphTensor) MaxPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	if gt.graph.State() != types.GraphBuilding {
		panic("gorgonia.GraphTensor.MaxPool2D: can only add operations during graph building")
	}

	node, err := gorgonia.MaxPool2D(gt.node, tensor.Shape{kernelSize[0], kernelSize[1]}, []int{padding[0], padding[1]}, []int{stride[0], stride[1]})
	if err != nil {
		panic(err)
	}

	// TODO: Calculate proper output shape
	outShape := gt.shape.Clone()

	return gt.graph.wrapNode(node, outShape, gt.dataType)
}

//
// Helper to wrap a node into a GraphTensor
//

func (eg *ExpressionGraph) wrapNode(node *gorgonia.Node, shape types.Shape, dataType types.DataType) *GraphTensor {
	gt := &GraphTensor{
		graph:    eg,
		node:     node,
		id:       eg.nextID,
		shape:    shape,
		dataType: dataType,
		constant: false,
	}

	eg.tensors = append(eg.tensors, gt)
	eg.nextID++

	return gt
}
