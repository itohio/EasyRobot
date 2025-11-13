package interpolation

import (
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// LinearMatrixUpsample upsamples a matrix using separable linear interpolation.
// dst must be pre-allocated with target rows and cols.
// Returns dst on success, nil on invalid input.
func LinearMatrixUpsample(src mat.Matrix, dst mat.Matrix) mat.Matrix {
	srcRows := len(src)
	if srcRows == 0 {
		return nil
	}

	srcCols := len(src[0])
	if srcCols == 0 {
		return nil
	}

	dstRows := len(dst)
	if dstRows == 0 {
		return nil
	}

	dstCols := len(dst[0])
	if dstCols == 0 {
		return nil
	}

	// Handle edge case: same size
	if srcRows == dstRows && srcCols == dstCols {
		return copyMatrix(src, dst)
	}

	// Handle edge case: 1x1 source
	if srcRows == 1 && srcCols == 1 {
		return fillMatrix(dst, src[0][0])
	}

	// Use separable interpolation: first upsample rows, then columns
	// Intermediate buffer for row-upsampled result
	intermediate := mat.New(srcRows, dstCols)
	if intermediate == nil {
		return nil
	}

	// Upsample each row
	srcRowVec := vec.New(srcCols)
	dstRowVec := vec.New(dstCols)

	for i := 0; i < srcRows; i++ {
		copy(srcRowVec, src[i])
		LinearUpsample(srcRowVec, dstRowVec)
		copy(intermediate[i], dstRowVec)
	}

	// Upsample each column of the intermediate result
	tempSrc := vec.New(srcRows)
	tempDst := vec.New(dstRows)

	for j := 0; j < dstCols; j++ {
		// Extract column
		for i := 0; i < srcRows; i++ {
			tempSrc[i] = intermediate[i][j]
		}

		// Upsample column
		LinearUpsample(tempSrc, tempDst)

		// Store column
		for i := 0; i < dstRows; i++ {
			dst[i][j] = tempDst[i]
		}
	}

	return dst
}

// BicubicMatrixUpsample upsamples a matrix using bicubic interpolation.
// dst must be pre-allocated with target rows and cols.
// Uses Mitchell-Netravali cubic kernel.
func BicubicMatrixUpsample(src mat.Matrix, dst mat.Matrix) mat.Matrix {
	srcRows := len(src)
	if srcRows == 0 {
		return nil
	}

	srcCols := len(src[0])
	if srcCols == 0 {
		return nil
	}

	dstRows := len(dst)
	if dstRows == 0 {
		return nil
	}

	dstCols := len(dst[0])
	if dstCols == 0 {
		return nil
	}

	// Handle edge case: same size
	if srcRows == dstRows && srcCols == dstCols {
		return copyMatrix(src, dst)
	}

	// Handle edge case: single pixel
	if srcRows == 1 && srcCols == 1 {
		return fillMatrix(dst, src[0][0])
	}

	// Scale factors
	rowScale := float32(srcRows-1) / float32(dstRows-1)
	colScale := float32(srcCols-1) / float32(dstCols-1)

	// Perform bicubic interpolation for each output pixel
	for i := 0; i < dstRows; i++ {
		for j := 0; j < dstCols; j++ {
			srcY := float32(i) * rowScale
			srcX := float32(j) * colScale

			dst[i][j] = bicubicInterpolate(src, srcX, srcY)
		}
	}

	return dst
}

// bicubicInterpolate performs bicubic interpolation at position (x, y) in source matrix.
func bicubicInterpolate(src mat.Matrix, x, y float32) float32 {
	// Get integer coordinates
	ix := int(x)
	iy := int(y)

	// Get fractional parts
	fx := x - float32(ix)
	fy := y - float32(iy)

	// Get 4x4 neighborhood with boundary handling
	var values [4][4]float32
	for dy := -1; dy <= 2; dy++ {
		for dx := -1; dx <= 2; dx++ {
			sx := ix + dx
			sy := iy + dy

			// Handle boundaries
			sx = clamp(sx, 0, len(src[0])-1)
			sy = clamp(sy, 0, len(src)-1)

			values[dy+1][dx+1] = src[sy][sx]
		}
	}

	// Compute bicubic interpolation
	// First, interpolate in x direction for each y
	var temp [4]float32
	for i := 0; i < 4; i++ {
		temp[i] = cubicInterpolate1D(values[i][0], values[i][1], values[i][2], values[i][3], fx)
	}

	// Then, interpolate in y direction
	result := cubicInterpolate1D(temp[0], temp[1], temp[2], temp[3], fy)

	return result
}

// cubicInterpolate1D performs 1D cubic interpolation at position t [0,1] using four points.
func cubicInterpolate1D(p0, p1, p2, p3, t float32) float32 {
	// Cubic interpolation using Catmull-Rom spline
	t2 := t * t
	t3 := t2 * t

	a0 := -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
	a1 := p0 - 2.5*p1 + 2*p2 - 0.5*p3
	a2 := -0.5*p0 + 0.5*p2

	return a0*t3 + a1*t2 + a2*t + p1
}

// Helper functions

func clamp(x, min, max int) int {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

func copyMatrix(src, dst mat.Matrix) mat.Matrix {
	for i := range src {
		copy(dst[i], src[i])
	}
	return dst
}

func fillMatrix(m mat.Matrix, value float32) mat.Matrix {
	for i := range m {
		for j := range m[i] {
			m[i][j] = value
		}
	}
	return m
}
