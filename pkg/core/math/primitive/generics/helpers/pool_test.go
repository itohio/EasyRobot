package helpers

import (
	"fmt"
	"sync"
	"testing"
)

func TestPool_DefaultConfig(t *testing.T) {
	var pool Pool[int]

	buf := pool.Get(12)
	if len(buf) != 12 {
		t.Fatalf("expected len 12, got %d", len(buf))
	}
	if cap(buf) != 16 {
		t.Fatalf("expected zeroth tier capacity 16, got %d", cap(buf))
	}

	pool.Put(buf)
}

func TestPool_ZerothTierReuse(t *testing.T) {
	pool := Pool[int]{Tiers: []int{64, 128}}

	buf := pool.Get(48)
	if len(buf) != 48 {
		t.Fatalf("expected len 48, got %d", len(buf))
	}
	if cap(buf) != 64 {
		t.Fatalf("expected cap 64 for zeroth tier, got %d", cap(buf))
	}

	firstPtr := &buf[:cap(buf)][0]
	pool.Put(buf)

	buf2 := pool.Get(32)
	if cap(buf2) != 64 {
		t.Fatalf("expected cap 64, got %d", cap(buf2))
	}

	secondPtr := &buf2[0]
	if firstPtr != secondPtr {
		t.Fatalf("expected buffer reuse within zeroth tier")
	}
}

func TestPool_PutAfterZeroLengthSlice(t *testing.T) {
	pool := Pool[int]{Tiers: []int{64}}

	buf := pool.Get(32)
	fullView := buf[:cap(buf)]
	firstPtr := &fullView[0]

	buf = buf[:0]
	pool.Put(buf)

	buf2 := pool.Get(16)
	if cap(buf2) != 64 {
		t.Fatalf("expected cap 64, got %d", cap(buf2))
	}
	secondPtr := &buf2[0]
	if firstPtr != secondPtr {
		t.Fatalf("expected buffer reuse after zero-length put")
	}
}

func TestPool_UnboundedTierReuse(t *testing.T) {
	pool := Pool[int]{Tiers: []int{64, 128, 512}}

	buf := pool.Get(1024)
	if len(buf) != 1024 {
		t.Fatalf("expected len 1024, got %d", len(buf))
	}
	if cap(buf) != 1024 {
		t.Fatalf("expected cap 1024 for unbounded tier, got %d", cap(buf))
	}

	firstPtr := &buf[:cap(buf)][0]
	pool.Put(buf)

	buf2 := pool.Get(1024)
	if cap(buf2) != 1024 {
		t.Fatalf("expected cap 1024, got %d", cap(buf2))
	}

	secondPtr := &buf2[0]
	if firstPtr != secondPtr {
		t.Fatalf("expected reuse in unbounded tier")
	}
}

func TestPool_Reconfigure(t *testing.T) {
	pool := Pool[int]{Tiers: []int{64, 128}}

	if err := pool.Reconfigure(32, 96); err != nil {
		t.Fatalf("unexpected reconfigure error: %v", err)
	}

	buf := pool.Get(24)
	if cap(buf) != 32 {
		t.Fatalf("expected cap 32 after reconfigure, got %d", cap(buf))
	}
	pool.Put(buf)

	buf2 := pool.Get(120)
	if cap(buf2) != 120 {
		t.Fatalf("expected exact cap for unbounded tier after reconfigure, got %d", cap(buf2))
	}
}

func TestPool_ReconfigureInvalid(t *testing.T) {
	pool := Pool[int]{Tiers: []int{32, 64}}

	if err := pool.Reconfigure(); err != nil {
		t.Fatalf("expected default reconfigure success, got error: %v", err)
	}

	if err := pool.Reconfigure(64, 32); err == nil {
		t.Fatalf("expected error for non-increasing tiers")
	}
}

func TestPool_ConcurrentGetPut(t *testing.T) {
	pool := Pool[int]{Tiers: []int{32, 128, 512}}

	const workers = 16
	const iterations = 1000

	var wg sync.WaitGroup
	wg.Add(workers)

	errCh := make(chan error, workers)

	for i := 0; i < workers; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				size := (j%600 + 1) + id
				buf := pool.Get(size)
				if len(buf) != size {
					errCh <- fmt.Errorf("worker %d iteration %d: expected len %d, got %d", id, j, size, len(buf))
					return
				}
				if cap(buf) < size {
					errCh <- fmt.Errorf("worker %d iteration %d: cap %d < size %d", id, j, cap(buf), size)
					return
				}
				pool.Put(buf)
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent get/put error: %v", err)
	}
}

func BenchmarkPoolGetPut(b *testing.B) {
	sizes := []int{8, 32, 64, 128, 1024}
	pool := Pool[int]{Tiers: []int{32, 128, 512, 2048}}

	b.ResetTimer()
	for _, size := range sizes {
		size := size
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				buf := pool.Get(size)
				if len(buf) != size {
					b.Fatalf("expected len %d, got %d", size, len(buf))
				}
				pool.Put(buf)
			}
		})
	}
}

func BenchmarkPoolUnbounded(b *testing.B) {
	pool := Pool[int]{Tiers: []int{32, 64, 128}}

	const size = 4096

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf := pool.Get(size)
		if len(buf) != size {
			b.Fatalf("expected len %d, got %d", size, len(buf))
		}
		pool.Put(buf)
	}
}
