package helpers

import (
	"errors"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

func newTestPool(tb testing.TB, pool *WorkerPool, opts ...WorkerPoolOption) *WorkerPool {
	tb.Helper()
	if err := pool.Init(opts...); err != nil {
		tb.Fatalf("Init failed: %v", err)
	}
	return pool
}

func TestWorkerPoolExecute(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 4}, WithTargetChunkSize(4))
	defer pool.Close()

	const total = 32
	visited := make([]int, total)

	err := pool.Execute(total, func(start, end int) error {
		for i := start; i < end; i++ {
			visited[i]++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	for i, count := range visited {
		if count != 1 {
			t.Fatalf("index %d processed %d times, want 1", i, count)
		}
	}
}

func TestWorkerPoolExecuteBeforeInit(t *testing.T) {
	t.Parallel()

	var pool WorkerPool
	err := pool.Execute(1, func(_, _ int) error { return nil })
	if !errors.Is(err, ErrPoolNotInitialized) {
		t.Fatalf("expected ErrPoolNotInitialized, got %v", err)
	}
}

func TestWorkerPoolExecuteError(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 3}, WithTargetChunkSize(2))
	defer pool.Close()

	wantErr := errors.New("boom")
	err := pool.Execute(6, func(start, end int) error {
		if start == 2 {
			return wantErr
		}
		return nil
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected error %v, got %v", wantErr, err)
	}
}

func TestWorkerPoolClose(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	pool.Close()

	err := pool.Execute(1, func(_, _ int) error { return nil })
	if !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("expected ErrPoolClosed, got %v", err)
	}
}

func TestWorkerPoolBlocksWhenNoWorkers(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 1}, WithTargetChunkSize(1))
	defer pool.Close()

	first := make(chan struct{})
	second := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{})

	go func() {
		_ = pool.Execute(2, func(start, end int) error {
			if start == 0 {
				close(first)
				<-release
			} else {
				close(second)
			}
			return nil
		})
		close(done)
	}()

	<-first
	select {
	case <-second:
		t.Fatal("second chunk started before first released worker")
	case <-time.After(50 * time.Millisecond):
	}

	close(release)
	<-second
	<-done
}

func TestWorkerPoolIteratorRangeEqualChunks(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2}, WithTargetChunkSize(4))
	defer pool.Close()

	const total = 16
	var got [][2]int
	for start, end := range pool.Iterator(total) {
		got = append(got, [2]int{start, end})
	}

	expectChunks := total / 4
	if len(got) != expectChunks {
		t.Fatalf("unexpected chunk count: got %d want %d", len(got), expectChunks)
	}
	for _, rng := range got {
		if rng[1]-rng[0] != 4 {
			t.Fatalf("expected chunk size 4, got %d", rng[1]-rng[0])
		}
	}
}

func TestWorkerPoolIteratorRangeUnequalChunks(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 3}, WithTargetChunkSize(5))
	defer pool.Close()

	const total = 18
	var got [][2]int
	for start, end := range pool.Iterator(total) {
		got = append(got, [2]int{start, end})
	}

	if len(got) != 4 {
		t.Fatalf("unexpected chunk count: got %d want %d", len(got), 4)
	}
	for idx, rng := range got {
		size := rng[1] - rng[0]
		if idx < len(got)-1 && size != 5 {
			t.Fatalf("chunk %d expected size 5, got %d", idx, size)
		}
		if idx == len(got)-1 && size != 3 {
			t.Fatalf("final chunk expected size 3, got %d", size)
		}
	}
}

func BenchmarkWorkerPool(b *testing.B) {
	const total = 1024
	pool := newTestPool(b, &WorkerPool{Size: 4}, WithTargetChunkSize(8))
	defer pool.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var mu sync.Mutex
		acc := 0
		if err := pool.Execute(total, func(start, end int) error {
			sum := 0
			for j := start; j < end; j++ {
				sum += j
			}
			mu.Lock()
			acc += sum
			mu.Unlock()
			return nil
		}); err != nil {
			b.Fatalf("Execute failed: %v", err)
		}
		if acc == 0 {
			b.Fatalf("unexpected accumulator value 0")
		}
	}
}

func BenchmarkSequentialSigmoid(b *testing.B) {
	const total = 256 * 1024
	inputs := make([]float64, total)
	outputs := make([]float64, total)
	for i := range inputs {
		inputs[i] = math.Sin(float64(i) * 0.01)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for idx, v := range inputs {
			outputs[idx] = sigmoid(v)
		}
	}
}

func BenchmarkWorkerPoolSigmoid(b *testing.B) {
	const total = 256 * 1024
	inputs := make([]float64, total)
	outputs := make([]float64, total)
	for i := range inputs {
		inputs[i] = math.Sin(float64(i) * 0.01)
	}

	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	chunk := total / workers
	if chunk < 1 {
		chunk = 1
	}

	pool := newTestPool(b, &WorkerPool{Size: workers}, WithTargetChunkSize(chunk))
	defer pool.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := pool.Execute(total, func(start, end int) error {
			for idx := start; idx < end; idx++ {
				outputs[idx] = sigmoid(inputs[idx])
			}
			return nil
		}); err != nil {
			b.Fatalf("Execute failed: %v", err)
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
