//go:build !use_mt

package workers

func ShouldParallelize(n int) bool {
	return false
}

func ParallelChunks(n int, fn func(start, end int)) {
	fn(0, n)
}

func ParallelRows(rows int, fn func(startRow, endRow int)) {
	fn(0, rows)
}

func ParallelTensorChunks(shape []int, fn func(startDim0, endDim0 int)) {

}

func ParallelIteratorChunks(totalSize int, fn func(startIdx, endIdx int)) {
	fn(0, totalSize)
}

func Execute(total int, fn func(start, end int)) {
	fn(0, total)
}
