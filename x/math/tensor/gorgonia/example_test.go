package gorgonia_test

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/gorgonia"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

func ExampleNew() {
	// Create a new FP32 tensor with shape [2, 3]
	t := gorgonia.New(types.FP32, 2, 3)

	fmt.Printf("Shape: %v\n", t.Shape())
	fmt.Printf("Rank: %d\n", t.Rank())
	fmt.Printf("Size: %d\n", t.Size())

	// Output:
	// Shape: [2 3]
	// Rank: 2
	// Size: 6
}

func ExampleTensor_Fill() {
	// Create and fill a tensor
	t := gorgonia.New(types.FP32, 2, 3)
	t = t.Fill(nil, 5.0).(gorgonia.Tensor)

	fmt.Printf("Element at [0,0]: %.1f\n", t.At(0, 0))
	fmt.Printf("Element at [1,2]: %.1f\n", t.At(1, 2))

	// Output:
	// Element at [0,0]: 5.0
	// Element at [1,2]: 5.0
}

func ExampleTensor_Add() {
	// Element-wise addition
	a := gorgonia.New(types.FP32, 2, 2)
	a.Fill(nil, 1.0)

	b := gorgonia.New(types.FP32, 2, 2)
	b.Fill(nil, 2.0)

	result := a.Add(nil, b).(gorgonia.Tensor)

	fmt.Printf("Result at [0,0]: %.1f\n", result.At(0, 0))
	fmt.Printf("Result at [1,1]: %.1f\n", result.At(1, 1))

	// Output:
	// Result at [0,0]: 3.0
	// Result at [1,1]: 3.0
}

func ExampleTensor_MatMul() {
	// Matrix multiplication
	a := gorgonia.New(types.FP32, 2, 3)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a.SetAt(float64(i*3+j+1), i, j)
		}
	}

	b := gorgonia.New(types.FP32, 3, 2)
	b.Fill(nil, 1.0)

	result := a.MatMul(nil, b).(gorgonia.Tensor)

	fmt.Printf("Result shape: %v\n", result.Shape())
	fmt.Printf("Result at [0,0]: %.1f\n", result.At(0, 0))
	fmt.Printf("Result at [1,0]: %.1f\n", result.At(1, 0))

	// Output:
	// Result shape: [2 2]
	// Result at [0,0]: 6.0
	// Result at [1,0]: 15.0
}

func ExampleTensor_Reshape() {
	// Reshape a tensor
	t := gorgonia.New(types.FP32, 2, 3)
	for i := 0; i < 6; i++ {
		t.SetAt(float64(i), i)
	}

	reshaped := t.Reshape(nil, []int{3, 2}).(gorgonia.Tensor)

	fmt.Printf("Original shape: %v\n", t.Shape())
	fmt.Printf("Reshaped shape: %v\n", reshaped.Shape())
	fmt.Printf("Data preserved: %.1f\n", reshaped.At(0))

	// Output:
	// Original shape: [2 3]
	// Reshaped shape: [3 2]
	// Data preserved: 0.0
}

func ExampleTensor_ScalarMul() {
	// Scalar multiplication
	t := gorgonia.New(types.FP32, 2, 2)
	t.Fill(nil, 3.0)

	result := t.ScalarMul(nil, 2.0).(gorgonia.Tensor)

	fmt.Printf("Result at [0,0]: %.1f\n", result.At(0, 0))
	fmt.Printf("Result at [1,1]: %.1f\n", result.At(1, 1))

	// Output:
	// Result at [0,0]: 6.0
	// Result at [1,1]: 6.0
}

func ExampleTensor_Copy() {
	// Demonstrate automatic conversion between tensor types using Copy
	
	// Create an eager tensor (from standard implementation)
	eagerTensor := gorgonia.New(types.FP32, 2, 2) // Using gorgonia for example simplicity
	eagerTensor.Fill(nil, 5.0)
	
	// Create a gorgonia tensor
	gorgoniaTensor := gorgonia.New(types.FP32, 2, 2)
	
	// Copy automatically handles conversion
	gorgoniaTensor.Copy(eagerTensor)
	
	fmt.Printf("Copied value at [0,0]: %.1f\n", gorgoniaTensor.At(0, 0))
	fmt.Printf("Copied value at [1,1]: %.1f\n", gorgoniaTensor.At(1, 1))
	
	// Output:
	// Copied value at [0,0]: 5.0
	// Copied value at [1,1]: 5.0
}

