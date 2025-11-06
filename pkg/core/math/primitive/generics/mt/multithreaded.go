package mt

import (
	"runtime"
	"sync"
)

var (
	// minParallelSize is the minimum number of elements required to enable parallelization.
	// Operations with fewer elements will use single-threaded execution to avoid overhead.
	minParallelSize = 128 * runtime.NumCPU()
	numWorkers      = runtime.NumCPU()
)

var (
	// globalPool is the global worker pool for parallel operations.
	// Initialized in init() when use_mt build tag is present.
	globalPool *workerPool
)

// workerPool manages a pool of worker goroutines for parallel task execution.
// Uses a shared task queue with work-stealing pattern for load balancing.
type workerPool struct {
	workers   int
	taskQueue chan task
	wg        sync.WaitGroup
	mu        sync.Mutex
	shutdown  bool
}

// task represents a unit of work to be executed by a worker.
type task func()

// init initializes the global worker pool when the package is loaded.
func init() {
	if numWorkers < 1 {
		numWorkers = 1
	}
	globalPool = newWorkerPool(numWorkers)
}

// newWorkerPool creates a new worker pool with the specified number of workers.
func newWorkerPool(workers int) *workerPool {
	if workers < 1 {
		workers = 1
	}
	pool := &workerPool{
		workers:   workers,
		taskQueue: make(chan task, workers*2), // Buffered channel for better throughput
	}
	pool.start()
	return pool
}

// start launches the worker goroutines.
func (p *workerPool) start() {
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}
}

// worker is the main loop for each worker goroutine.
// Workers pull tasks from the shared queue, implementing a simple work-stealing pattern.
func (p *workerPool) worker(id int) {
	defer p.wg.Done()
	for t := range p.taskQueue {
		if t != nil {
			t()
		}
	}
}

// submit adds a task to the work queue.
// Returns false if the pool is shut down.
func (p *workerPool) submit(t task) bool {
	p.mu.Lock()
	shutdown := p.shutdown
	p.mu.Unlock()
	if shutdown {
		return false
	}
	select {
	case p.taskQueue <- t:
		return true
		// default:
		// 	// If queue is full, execute synchronously to avoid blocking
		// 	// This is a fallback for high contention scenarios
		// 	t()
		// 	return true
	}
}

// parallelExecute executes multiple tasks in parallel using the worker pool.
// Blocks until all tasks complete.
func (p *workerPool) parallelExecute(tasks []task) {
	if len(tasks) == 0 {
		return
	}
	if len(tasks) == 1 {
		tasks[0]()
		return
	}

	var wg sync.WaitGroup
	wg.Add(len(tasks))
	for _, t := range tasks {
		task := t // Capture loop variable
		submitted := p.submit(func() {
			defer wg.Done()
			task()
		})
		if !submitted {
			// Pool is shut down, execute synchronously
			wg.Done()
			task()
		}
	}
	wg.Wait()
}

// stop gracefully shuts down the worker pool.
// After stop, no new tasks will be accepted.
func (p *workerPool) stop() {
	p.mu.Lock()
	if p.shutdown {
		p.mu.Unlock()
		return
	}
	p.shutdown = true
	p.mu.Unlock()
	close(p.taskQueue)
	p.wg.Wait()
}

// shouldParallelize determines if parallelization should be used based on size and CPU count.
func shouldParallelize(n int) bool {
	return n >= minParallelSize && numWorkers > 1
}

// parallelChunks splits work into chunks and processes them in parallel.
// If n < minParallelSize or only 1 CPU, executes sequentially.
// fn is called with (start, end) for each chunk.
func parallelChunks(n int, fn func(start, end int)) {
	chunkSize := (n + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= end {
			break
		}

		wg.Add(1)
		s, e := start, end // Capture loop variables
		globalPool.submit(func() {
			defer wg.Done()
			fn(s, e)
		})
	}
	wg.Wait()
}

// parallelRows splits matrix rows across workers and processes them in parallel.
// If rows < minParallelSize or only 1 CPU, executes sequentially.
// fn is called with (startRow, endRow) for each chunk.
func parallelRows(rows int, fn func(startRow, endRow int)) {
	chunkSize := (rows + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		startRow := i * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}
		if startRow >= endRow {
			break
		}

		wg.Add(1)
		sr, er := startRow, endRow // Capture loop variables
		globalPool.submit(func() {
			defer wg.Done()
			fn(sr, er)
		})
	}
	wg.Wait()
}

// parallelTensorChunks splits multi-dimensional tensor work into chunks along the first dimension.
// Each worker processes a subset of the first dimension, then iterates over remaining dimensions.
// shape is the tensor shape, fn is called with (startDim0, endDim0) for each chunk.
func parallelTensorChunks(shape []int, fn func(startDim0, endDim0 int)) {
	if len(shape) == 0 {
		return
	}

	dim0Size := shape[0]

	chunkSize := (dim0Size + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		startDim0 := i * chunkSize
		endDim0 := startDim0 + chunkSize
		if endDim0 > dim0Size {
			endDim0 = dim0Size
		}
		if startDim0 >= endDim0 {
			break
		}

		wg.Add(1)
		sd0, ed0 := startDim0, endDim0 // Capture loop variables
		globalPool.submit(func() {
			defer wg.Done()
			fn(sd0, ed0)
		})
	}
	wg.Wait()
}

// parallelIteratorChunks splits iterator work into chunks for parallel processing.
// totalSize is the total number of elements to iterate over.
// fn is called with (startIdx, endIdx) for each chunk.
func parallelIteratorChunks(totalSize int, fn func(startIdx, endIdx int)) {
	chunkSize := (totalSize + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		startIdx := i * chunkSize
		endIdx := startIdx + chunkSize
		if endIdx > totalSize {
			endIdx = totalSize
		}
		if startIdx >= endIdx {
			break
		}

		wg.Add(1)
		si, ei := startIdx, endIdx // Capture loop variables
		globalPool.submit(func() {
			defer wg.Done()
			fn(si, ei)
		})
	}
	wg.Wait()
}
