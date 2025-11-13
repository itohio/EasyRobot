// Package gorgonia provides a lightweight wrapper around gorgonia.org/tensor
// that implements the EasyRobot tensor types.Tensor interface.
//
// This package serves as a bridge between gorgonia's tensor operations
// and EasyRobot's tensor abstraction layer, enabling interoperability
// between the two systems while maintaining the interface contract.
//
// All tensor methods use value receivers to match the types.Tensor interface
// requirements and ensure consistent behavior across implementations.
//
// Example usage:
//
//	import "github.com/itohio/EasyRobot/x/math/tensor/gorgonia"
//
//	// Create a new tensor with shape [2, 3, 4]
//	t := gorgonia.New(2, 3, 4)
//
//	// Use tensor operations
//	t.Fill(nil, 1.0)
//	sum := t.Sum(nil, nil) // Sum all elements
package gorgonia
