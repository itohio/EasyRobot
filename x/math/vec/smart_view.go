package vec

import (
	"sync/atomic"

	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// refCount is a shared atomic reference counter used by all views of the same object.
type refCount struct {
	count int64
}

// SmartVector wraps a Vector with reference counting.
// It embeds Vector interface to promote all methods, allowing direct calls like smartA.MulC(...).
//
// Example usage:
//
//	smartA := vec.NewSmart(3)  // Same signature as vec.New()
//	a1 := smartA.View()
//	a2 := smartA.View()
//	a1.MulC(...)  // Direct method call - no .Value() needed!
//	a1.Release()
//	a2.Release() // Last reference released, underlying vector is released
type SmartVector struct {
	vecTypes.Vector
	refs *refCount
}

// NewSmart creates a new SmartVector with reference counting.
// It accepts the same parameters as New() for consistency.
// The initial reference count is 1.
func NewSmart(size int) *SmartVector {
	return &SmartVector{
		Vector: New(size),
		refs:   &refCount{count: 1},
	}
}

// WithRefcount wraps an existing Vector with reference counting.
// The initial reference count is 1.
func WithRefcount(v vecTypes.Vector) *SmartVector {
	return &SmartVector{
		Vector: v,
		refs:   &refCount{count: 1},
	}
}

// View creates a new view of the wrapped vector.
// The view implements the Vector interface and can be used directly.
func (s *SmartVector) View() *VectorView {
	atomic.AddInt64(&s.refs.count, 1)
	return &VectorView{
		Vector: s.Vector,
		refs:   s.refs,
	}
}

// Release decrements the reference count and releases the wrapped vector when count reaches zero.
// Safe to call multiple times (idempotent).
func (s *SmartVector) Release() {
	if s.refs == nil {
		return // already released
	}
	if atomic.AddInt64(&s.refs.count, -1) == 0 {
		s.Vector.Release()
		s.refs = nil
	}
}

// VectorView is an independent view of a SmartVector.
// It embeds Vector interface to promote all methods, allowing direct calls like view.MulC(...).
type VectorView struct {
	vecTypes.Vector
	refs *refCount
}

// Release decrements the reference count and releases the wrapped vector when count reaches zero.
// Safe to call multiple times (idempotent).
func (v *VectorView) Release() {
	if v.refs == nil {
		return // already released
	}
	if atomic.AddInt64(&v.refs.count, -1) == 0 {
		v.Vector.Release()
		v.refs = nil
	}
}

// Note: SmartVector and VectorView shadow the View() method from the Vector interface,
// so they don't fully satisfy the Vector interface. This is intentional - they are
// wrapper types that provide reference counting. All other Vector methods are promoted
// via embedding and can be called directly (e.g., smartA.MulC(...)).
