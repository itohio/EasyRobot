package helpers

import (
	"fmt"
	"sync"
)

const (
	defaultTierCount  = 8
	defaultTierStart  = 16
	defaultTierFactor = 2
)

// Pool provides typed buffer reuse across several capacity tiers.
// It embeds sync.RWMutex to guard configuration and ensures tiered slices are not copied concurrently.
type Pool[T any] struct {
	sync.RWMutex
	Tiers     []int
	bounds    []int
	tierPools []sync.Pool // len(bounds)+1, last one is unbounded
}

// Reconfigure replaces the pool's tier configuration.
func (p *Pool[T]) Reconfigure(lengths ...int) error {
	p.Lock()
	defer p.Unlock()

	if err := p.configureLocked(lengths...); err != nil {
		return err
	}

	if len(lengths) == 0 {
		p.Tiers = append([]int(nil), p.bounds...)
	} else {
		p.Tiers = append([]int(nil), lengths...)
	}

	return nil
}

// Get returns a buffer of length n with capacity within the configured tiers.
// IMPORTANT: Get DOES NOT guarantee initialized buffers! Use Fill() to initialize the buffer if necessary.
func (p *Pool[T]) Get(n int) []T {
	p.RLock()
	if len(p.bounds) == 0 {
		p.RUnlock()
		p.ensureDefault()
		p.RLock()
	}

	idx := tierIndex(p.bounds, n)
	pool := &p.tierPools[idx]
	raw := pool.Get()
	capacity := capacityFor(p.bounds, idx, n)
	p.RUnlock()

	var buf []T
	if raw != nil {
		buf = raw.([]T)
	}
	if buf == nil || cap(buf) < n {
		buf = make([]T, 0, capacity)
	}

	return buf[:n]
}

// Put returns buffer to appropriate tier.
func (p *Pool[T]) Put(buf []T) {
	if buf == nil {
		return
	}

	length := cap(buf)
	if length == 0 {
		return
	}

	p.RLock()
	if len(p.bounds) == 0 {
		p.RUnlock()
		p.ensureDefault()
		p.RLock()
	}
	idx := tierIndex(p.bounds, length)
	pool := &p.tierPools[idx]
	p.RUnlock()

	pool.Put(buf[:0])
}

func (p *Pool[T]) ensureDefault() {
	p.Lock()
	defer p.Unlock()

	if len(p.bounds) != 0 {
		return
	}
	if err := p.configureLocked(p.Tiers...); err != nil {
		panic(err)
	}
	if len(p.Tiers) == 0 {
		p.Tiers = append([]int(nil), p.bounds...)
	}
}

func (p *Pool[T]) configureLocked(lengths ...int) error {
	if len(lengths) == 0 {
		lengths = defaultTierLengths()
	}

	bounds := make([]int, len(lengths))
	var prev int
	for i, maxLen := range lengths {
		if maxLen <= 0 {
			return fmt.Errorf("generics: tier %d has non-positive length %d", i, maxLen)
		}
		if i > 0 && maxLen <= prev {
			return fmt.Errorf("generics: tier %d length %d must be greater than previous %d", i, maxLen, prev)
		}
		bounds[i] = maxLen
		prev = maxLen
	}

	tierPools := make([]sync.Pool, len(bounds)+1)
	for i := range tierPools {
		if i < len(bounds) {
			maxLen := bounds[i]
			tierPools[i].New = makeTierFactory[T](maxLen)
		}
	}

	p.bounds = bounds
	p.tierPools = tierPools
	return nil
}

func defaultTierLengths() []int {
	lengths := make([]int, defaultTierCount)
	value := defaultTierStart
	for i := 0; i < defaultTierCount; i++ {
		lengths[i] = value
		value *= defaultTierFactor
	}
	return lengths
}

func tierIndex(bounds []int, length int) int {
	for i, bound := range bounds {
		if length <= bound {
			return i
		}
	}
	return len(bounds)
}

func capacityFor(bounds []int, idx, n int) int {
	if idx < len(bounds) {
		maxLen := bounds[idx]
		if n > maxLen {
			return n
		}
		return maxLen
	}
	return n
}

func makeTierFactory[T any](maxLen int) func() any {
	return func() any {
		return make([]T, 0, maxLen)
	}
}
