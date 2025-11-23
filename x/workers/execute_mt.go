//go:build use_mt

package workers

func ShouldParallelize(n int) bool {
	mt.ShouldParallelize(n)
}

func ParallelChunks(n int, fn func(start, end int)) {
	mt.ParallelChunks(n, fn)
}

func ParallelRows(rows int, fn func(startRow, endRow int)) {
	mt.ParallelRows(rows, fn)
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
	mt.ParallelIteratorChunks(totalSize, fn)
}

func Execute(total int, fn func(start, end int)) {
	return mt.Execute(total, fn)
}
