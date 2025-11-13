package mt

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

var (
	minParallelSize = 128 * runtime.NumCPU()
	numWorkers      = runtime.NumCPU()
)

var (
	poolOnce sync.Once
	poolErr  error
	pool     *helpers.WorkerPool
)

func init() {
	initWorkerPool()
}

func initWorkerPool() {
	poolOnce.Do(func() {
		if numWorkers < 1 {
			numWorkers = 1
		}
		minParallelSize = 128 * numWorkers

		w := &helpers.WorkerPool{
			Size: numWorkers,
		}
		poolErr = w.Init()
		if poolErr != nil {
			return
		}
		pool = w
	})
	if poolErr != nil {
		panic(fmt.Errorf("generics/mt: initializing worker pool: %w", poolErr))
	}
}

func workerPool() *helpers.WorkerPool {
	if pool == nil {
		initWorkerPool()
	}
	return pool
}

func shouldParallelize(n int) bool {
	return n >= minParallelSize && numWorkers > 1
}

func parallelChunks(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	executeParallel(n, fn)
}

func parallelRows(rows int, fn func(startRow, endRow int)) {
	if rows <= 0 {
		return
	}
	executeParallel(rows, fn)
}

func parallelTensorChunks(shape []int, fn func(startDim0, endDim0 int)) {
	if len(shape) == 0 {
		return
	}
	dim0Size := shape[0]
	if dim0Size <= 0 {
		return
	}
	executeParallel(dim0Size, fn)
}

func parallelIteratorChunks(totalSize int, fn func(startIdx, endIdx int)) {
	if totalSize <= 0 {
		return
	}
	executeParallel(totalSize, fn)
}

func executeParallel(total int, fn func(start, end int)) {
	p := workerPool()
	if p == nil {
		panic("generics/mt: worker pool not initialised")
	}
	if err := p.Execute(total, func(start, end int) error {
		fn(start, end)
		return nil
	}); err != nil {
		panic(fmt.Errorf("generics/mt: executing worker pool job: %w", err))
	}
}
