package mt

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/itohio/EasyRobot/x/workers"
)

var (
	minParallelSize = 128 * runtime.NumCPU()
	numWorkers      = runtime.NumCPU()
)

var (
	poolOnce sync.Once
	poolErr  error
	pool     workers.WorkerPool
)

func init() {
	poolOnce.Do(func() {
		if numWorkers < 1 {
			numWorkers = 1
		}
		minParallelSize = 128 * numWorkers

		pool.Size = numWorkers
		poolErr = pool.Init()
		if poolErr != nil {
			return
		}
	})
	if poolErr != nil {
		panic(fmt.Errorf("generics/mt: initializing worker pool: %w", poolErr))
	}
}

func ShouldParallelize(n int) bool {
	return n >= minParallelSize && numWorkers > 1
}

func ParallelChunks(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	Execute(n, fn)
}

func ParallelRows(rows int, fn func(startRow, endRow int)) {
	if rows <= 0 {
		return
	}
	Execute(rows, fn)
}

func ParallelTensorChunks(shape []int, fn func(startDim0, endDim0 int)) {
	if len(shape) == 0 {
		return
	}
	dim0Size := shape[0]
	if dim0Size <= 0 {
		return
	}
	Execute(dim0Size, fn)
}

func ParallelIteratorChunks(totalSize int, fn func(startIdx, endIdx int)) {
	if totalSize <= 0 {
		return
	}
	Execute(totalSize, fn)
}

func Execute(total int, fn func(start, end int)) {
	if err := pool.Execute(total, func(start, end int) error {
		fn(start, end)
		return nil
	}); err != nil {
		panic(fmt.Errorf("generics/mt: executing worker pool job: %w", err))
	}
}
