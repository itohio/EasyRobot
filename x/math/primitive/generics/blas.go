package generics

// Copy copies x into y for contiguous arrays.
// This is BLAS COPY operation optimized for the common case of contiguous memory.
func Copy[T Numeric](y, x []T, n int) {
	if n == 0 {
		return
	}
	copy(y[:n], x[:n])
}

// CopyStrided copies x into y with stride support.
// This is BLAS COPY operation with stride parameters.
func CopyStrided[T Numeric](y, x []T, strideY, strideX, n int) {
	if n == 0 {
		return
	}

	if strideY == 1 && strideX == 1 {
		// Fast path: contiguous arrays
		copy(y[:n], x[:n])
		return
	}

	// Strided path
	py := 0
	px := 0
	for i := 0; i < n; i++ {
		y[py] = x[px]
		py += strideY
		px += strideX
	}
}

// Swap swaps x and y for contiguous arrays.
// This is BLAS SWAP operation optimized for the common case of contiguous memory.
func Swap[T Numeric](x, y []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}

// SwapStrided swaps x and y with stride support.
// This is BLAS SWAP operation with stride parameters.
func SwapStrided[T Numeric](x, y []T, strideX, strideY, n int) {
	if n == 0 {
		return
	}

	if strideX == 1 && strideY == 1 {
		// Fast path: contiguous arrays
		for i := 0; i < n; i++ {
			x[i], y[i] = y[i], x[i]
		}
		return
	}

	// Strided path
	px := 0
	py := 0
	for i := 0; i < n; i++ {
		x[px], y[py] = y[py], x[px]
		px += strideX
		py += strideY
	}
}
