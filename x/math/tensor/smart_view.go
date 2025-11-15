package tensor

import (
	"sync/atomic"

	tensorTypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// refCount is a shared atomic reference counter used by all views of the same object.
type refCount struct {
	count int64
}

// SmartTensor wraps a Tensor with reference counting.
// It embeds Tensor interface to promote all methods, allowing direct calls like smartA.Mul(...).
//
// Example usage:
//
//	smartA := tensor.NewSmart(tensor.DTFP32, tensor.Shape{2, 3})  // Same signature as tensor.New()
//	// Or: smartA := tensor.NewSmart(tensor.DTFP32, []int{2, 3})
//	a1 := smartA.View()
//	a2 := smartA.View()
//	a1.Mul(...)  // Direct method call - no .Value() needed!
//	a1.Release()
//	a2.Release() // Last reference released, underlying tensor is released
type SmartTensor struct {
	tensorTypes.Tensor
	refs *refCount
}

// NewSmart creates a new SmartTensor with reference counting.
// It accepts the same parameters as New() for consistency.
// The initial reference count is 1.
func NewSmart(dtype DataType, shape Shape) *SmartTensor {
	return &SmartTensor{
		Tensor: New(dtype, shape),
		refs:   &refCount{count: 1},
	}
}

// WithRefcount wraps an existing Tensor with reference counting.
// The initial reference count is 1.
func WithRefcount(t tensorTypes.Tensor) *SmartTensor {
	return &SmartTensor{
		Tensor: t,
		refs:   &refCount{count: 1},
	}
}

// View creates a new view of the wrapped tensor.
// The view implements the Tensor interface and can be used directly.
func (s *SmartTensor) View() *TensorView {
	atomic.AddInt64(&s.refs.count, 1)
	return &TensorView{
		Tensor: s.Tensor,
		refs:   s.refs,
	}
}

// Release decrements the reference count and releases the wrapped tensor when count reaches zero.
// Safe to call multiple times (idempotent).
func (s *SmartTensor) Release() {
	if s.refs == nil {
		return // already released
	}
	if atomic.AddInt64(&s.refs.count, -1) == 0 {
		s.Tensor.Release()
		s.refs = nil
	}
}

// TensorView is an independent view of a SmartTensor.
// It embeds Tensor interface to promote all methods, allowing direct calls like view.Mul(...).
type TensorView struct {
	tensorTypes.Tensor
	refs *refCount
}

// Release decrements the reference count and releases the wrapped tensor when count reaches zero.
// Safe to call multiple times (idempotent).
func (v *TensorView) Release() {
	if v.refs == nil {
		return // already released
	}
	if atomic.AddInt64(&v.refs.count, -1) == 0 {
		v.Tensor.Release()
		v.refs = nil
	}
}

// Note: SmartTensor and TensorView shadow the View() method from the Tensor interface,
// so they don't fully satisfy the Tensor interface. This is intentional - they are
// wrapper types that provide reference counting. All other Tensor methods are promoted
// via embedding and can be called directly (e.g., smartA.Mul(...)).
