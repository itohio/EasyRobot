package mat

import (
	"sync/atomic"

	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

// refCount is a shared atomic reference counter used by all views of the same object.
type refCount struct {
	count int64
}

// SmartMatrix wraps a Matrix with reference counting.
// It embeds Matrix interface to promote all methods, allowing direct calls like smartA.Mul(...).
//
// Example usage:
//
//	smartA := mat.NewSmart(3, 3)  // Same signature as mat.New()
//	a1 := smartA.View()
//	a2 := smartA.View()
//	a1.Mul(...)  // Direct method call - no .Value() needed!
//	a1.Release()
//	a2.Release() // Last reference released, underlying matrix is released
type SmartMatrix struct {
	matTypes.Matrix
	refs *refCount
}

// NewSmart creates a new SmartMatrix with reference counting.
// It accepts the same parameters as New() for consistency.
// The initial reference count is 1.
func NewSmart(rows, cols int, backing ...float32) *SmartMatrix {
	return &SmartMatrix{
		Matrix: New(rows, cols, backing...),
		refs:   &refCount{count: 1},
	}
}

// WithRefcount wraps an existing Matrix with reference counting.
// The initial reference count is 1.
func WithRefcount(m matTypes.Matrix) *SmartMatrix {
	return &SmartMatrix{
		Matrix: m,
		refs:   &refCount{count: 1},
	}
}

// View creates a new view of the wrapped matrix.
// The view implements the Matrix interface and can be used directly.
func (s *SmartMatrix) View() *MatrixView {
	atomic.AddInt64(&s.refs.count, 1)
	return &MatrixView{
		Matrix: s.Matrix,
		refs:   s.refs,
	}
}

// Release decrements the reference count and releases the wrapped matrix when count reaches zero.
// Safe to call multiple times (idempotent).
func (s *SmartMatrix) Release() {
	if s.refs == nil {
		return // already released
	}
	if atomic.AddInt64(&s.refs.count, -1) == 0 {
		s.Matrix.Release()
		s.refs = nil
	}
}

// MatrixView is an independent view of a SmartMatrix.
// It embeds Matrix interface to promote all methods, allowing direct calls like view.Mul(...).
type MatrixView struct {
	matTypes.Matrix
	refs *refCount
}

// Release decrements the reference count and releases the wrapped matrix when count reaches zero.
// Safe to call multiple times (idempotent).
func (v *MatrixView) Release() {
	if v.refs == nil {
		return // already released
	}
	if atomic.AddInt64(&v.refs.count, -1) == 0 {
		v.Matrix.Release()
		v.refs = nil
	}
}

// Note: SmartMatrix and MatrixView shadow the View() method from the Matrix interface,
// so they don't fully satisfy the Matrix interface. This is intentional - they are
// wrapper types that provide reference counting. All other Matrix methods are promoted
// via embedding and can be called directly (e.g., smartA.Mul(...)).
