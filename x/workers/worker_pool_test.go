package workers

import (
	"context"
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

func TestWorkerPoolDo(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 4})
	defer pool.Close()

	const numCallbacks = 10
	executed := make([]bool, numCallbacks)
	var mu sync.Mutex

	callbacks := make([]SimpleCallback, numCallbacks)
	for i := range callbacks {
		idx := i
		callbacks[i] = func(ctx context.Context) error {
			mu.Lock()
			executed[idx] = true
			mu.Unlock()
			return nil
		}
	}

	ctx := context.Background()
	err := pool.Do(ctx, callbacks...)
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}

	for i, exec := range executed {
		if !exec {
			t.Fatalf("callback %d was not executed", i)
		}
	}
}

func TestWorkerPoolDoBeforeInit(t *testing.T) {
	t.Parallel()

	var pool WorkerPool
	ctx := context.Background()
	err := pool.Do(ctx, func(ctx context.Context) error { return nil })
	if !errors.Is(err, ErrPoolNotInitialized) {
		t.Fatalf("expected ErrPoolNotInitialized, got %v", err)
	}
}

func TestWorkerPoolDoNilContext(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	err := pool.Do(nil, func(ctx context.Context) error { return nil })
	if err == nil {
		t.Fatal("expected error for nil context, got nil")
	}
}

func TestWorkerPoolDoAfterClose(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	pool.Close()

	ctx := context.Background()
	err := pool.Do(ctx, func(ctx context.Context) error { return nil })
	if !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("expected ErrPoolClosed, got %v", err)
	}
}

func TestWorkerPoolDoEmptyCallbacks(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	ctx := context.Background()
	err := pool.Do(ctx)
	if err != nil {
		t.Fatalf("expected nil error for empty callbacks, got %v", err)
	}
}

func TestWorkerPoolDoNilCallbacks(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	executed := false
	ctx := context.Background()
	err := pool.Do(ctx,
		nil,
		func(ctx context.Context) error {
			executed = true
			return nil
		},
		nil,
	)
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}
	if !executed {
		t.Fatal("non-nil callback was not executed")
	}
}

func TestWorkerPoolDoError(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 3})
	defer pool.Close()

	wantErr := errors.New("callback error")
	ctx := context.Background()
	err := pool.Do(ctx,
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return wantErr },
		func(ctx context.Context) error { return nil },
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected error %v, got %v", wantErr, err)
	}
}

func TestWorkerPoolDoContextCancellation(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	done := make(chan struct{})

	go func() {
		err := pool.Do(ctx,
			func(ctx context.Context) error {
				close(started)
				time.Sleep(100 * time.Millisecond)
				return nil
			},
			func(ctx context.Context) error {
				time.Sleep(100 * time.Millisecond)
				return nil
			},
		)
		if err == nil {
			t.Error("expected context cancellation error, got nil")
		}
		close(done)
	}()

	<-started
	cancel()
	<-done
}

func TestWorkerPoolDoContextTimeout(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := pool.Do(ctx,
		func(ctx context.Context) error {
			time.Sleep(100 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			time.Sleep(100 * time.Millisecond)
			return nil
		},
	)
	if err == nil {
		t.Fatal("expected context timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
}

func TestWorkerPoolDoParallelExecution(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	const numCallbacks = 5
	order := make([]int, 0, numCallbacks)
	var mu sync.Mutex
	barrier := make(chan struct{})

	callbacks := make([]SimpleCallback, numCallbacks)
	for i := range callbacks {
		idx := i
		callbacks[i] = func(ctx context.Context) error {
			<-barrier
			mu.Lock()
			order = append(order, idx)
			mu.Unlock()
			return nil
		}
	}

	ctx := context.Background()
	go func() {
		time.Sleep(10 * time.Millisecond)
		for range numCallbacks {
			barrier <- struct{}{}
		}
	}()

	err := pool.Do(ctx, callbacks...)
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}

	if len(order) != numCallbacks {
		t.Fatalf("expected %d callbacks executed, got %d", numCallbacks, len(order))
	}
}

func TestWorkerPoolDoContextPassedToCallbacks(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 2})
	defer pool.Close()

	ctx := context.WithValue(context.Background(), "test-key", "test-value")
	receivedCtx := make(chan context.Context, 1)

	err := pool.Do(ctx, func(cbCtx context.Context) error {
		receivedCtx <- cbCtx
		return nil
	})
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}

	cbCtx := <-receivedCtx
	val := cbCtx.Value("test-key")
	if val != "test-value" {
		t.Fatalf("expected context value 'test-value', got %v", val)
	}
}

func TestWorkerPoolDoBlocksWhenNoWorkers(t *testing.T) {
	t.Parallel()

	pool := newTestPool(t, &WorkerPool{Size: 1})
	defer pool.Close()

	first := make(chan struct{})
	second := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{})

	ctx := context.Background()
	go func() {
		_ = pool.Do(ctx,
			func(ctx context.Context) error {
				close(first)
				<-release
				return nil
			},
			func(ctx context.Context) error {
				close(second)
				return nil
			},
		)
		close(done)
	}()

	<-first
	select {
	case <-second:
		t.Fatal("second callback started before first released worker")
	case <-time.After(50 * time.Millisecond):
	}

	close(release)
	<-second
	<-done
}

func BenchmarkWorkerPoolDo(b *testing.B) {
	pool := newTestPool(b, &WorkerPool{Size: 4})
	defer pool.Close()

	ctx := context.Background()
	callbacks := make([]SimpleCallback, 100)
	for i := range callbacks {
		callbacks[i] = func(ctx context.Context) error {
			return nil
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := pool.Do(ctx, callbacks...); err != nil {
			b.Fatalf("Do failed: %v", err)
		}
	}
}
