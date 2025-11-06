package types

// RNG interface for random number generation used in tensor operations.
// This interface is implemented by types like *rand.Rand from math/rand.
type RNG interface {
	Float64() float64
	NormFloat64() float64
}

// Element interface represents a single tensor element with Get and Set methods.
type Element interface {
	Get() float64
	Set(value float64)
}

// Tensor defines the complete interface for tensor operations.
// This interface is the authoritative source of truth for all Tensor capabilities.
// All concrete implementations must satisfy this interface.
// Concrete implementations must be using Value receivers instead of pointers.
//
// The Tensor interface is composed of category interfaces for better organization.
// Operations are gradually being moved from this interface to category interfaces.
type Tensor interface {
	// Core Properties and Access
	// TensorCore interface is embedded to provide core tensor operations.
	TensorCore

	// TensorManipulation interface is embedded to provide tensor manipulation operations.
	TensorManipulation

	// TensorElementWise interface is embedded to provide element-wise operations.
	TensorElementWise

	// TensorMath interface is embedded to provide math operations (reductions, linear algebra).
	TensorMath

	// TensorActivations interface is embedded to provide activation functions.
	TensorActivations

	// TensorConvolutions interface is embedded to provide convolution operations.
	TensorConvolutions

	// TensorPooling interface is embedded to provide pooling operations.
	TensorPooling

	// TensorDropout interface is embedded to provide dropout operations.
	TensorDropout
}

// Helper functions that would accept Tensor interface:
// - ZerosLike(t Tensor) Tensor
// - OnesLike(t Tensor) Tensor
// - FullLike(t Tensor, value float32) Tensor
//
// NOTE: These helper functions currently return *Tensor (pointer).
// With interface-based design, they should return Tensor interface.
// However, they would need to create concrete EagerTensor instances internally.
//
// NOTE on DropoutMask: The rng parameter uses the RNG interface defined in this package,
// which is implemented by types like *rand.Rand from math/rand.

// Must is a helper function that panics if the error is not nil.
// It is used to simplify error handling in the code.
// It is similar to the Must function in the standard library, but it is
// more general and can be used with any type.
func Must(t any, err error) any {
	if err != nil {
		panic(err)
	}
	return t
}
