package grid

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
)

// Mask masks one matrix with another (element-wise multiplication).
// For each element: result[i][j] = src[i][j] * mask[i][j]
// If mask is nil or has different dimensions, returns src unchanged.
//
// Parameters:
//   - src: Source matrix
//   - mask: Mask matrix (same dimensions as src, values typically 0.0 or 1.0)
//   - dst: Destination matrix (can be same as src for in-place operation)
//
// Returns dst (or src if mask invalid).
func Mask(src, mask, dst mat.Matrix) mat.Matrix {
	if mask == nil || len(mask) != len(src) || len(mask[0]) != len(src[0]) {
		return src
	}

	if dst == nil || len(dst) != len(src) || len(dst[0]) != len(src[0]) {
		dst = mat.New(len(src), len(src[0]))
	}

	rows := len(src)
	cols := len(src[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dst[i][j] = src[i][j] * mask[i][j]
		}
	}

	return dst
}

// ExtractRectangle extracts a rectangular region from a matrix.
//
// Parameters:
//   - src: Source matrix
//   - x, y: Top-left corner of rectangle (in matrix coordinates)
//   - width, height: Width and height of rectangle
//
// Returns extracted rectangle matrix, or nil if bounds invalid.
func ExtractRectangle(src mat.Matrix, x, y, width, height int) mat.Matrix {
	rows := len(src)
	cols := len(src[0])

	// Check bounds
	if x < 0 || y < 0 || width <= 0 || height <= 0 {
		return nil
	}
	if x+width > cols || y+height > rows {
		return nil
	}

	rect := mat.New(height, width)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			rect[i][j] = src[y+i][x+j]
		}
	}

	return rect
}

// ExtractCircle extracts a circular region from a matrix.
//
// Parameters:
//   - src: Source matrix
//   - centerX, centerY: Center of circle (in matrix coordinates)
//   - radius: Radius of circle (in grid cells)
//   - fillValue: Value to use for cells outside circle (default: 0.0)
//
// Returns extracted circle matrix, or nil if bounds invalid.
func ExtractCircle(src mat.Matrix, centerX, centerY, radius int, fillValue float32) mat.Matrix {
	rows := len(src)
	cols := len(src[0])

	// Check bounds
	if centerX-radius < 0 || centerX+radius >= cols ||
		centerY-radius < 0 || centerY+radius >= rows {
		return nil
	}

	if radius <= 0 {
		return nil
	}

	size := 2*radius + 1
	circle := mat.New(size, size)

	// Initialize with fill value
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			circle[i][j] = fillValue
		}
	}

	radiusSq := float32(radius) * float32(radius)

	// Extract circular region
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			// Calculate distance from center
			dx := float32(j - radius)
			dy := float32(i - radius)
			distSq := dx*dx + dy*dy

			// If within circle, copy from source
			if distSq <= radiusSq {
				srcY := centerY - radius + i
				srcX := centerX - radius + j
				if srcY >= 0 && srcY < rows && srcX >= 0 && srcX < cols {
					circle[i][j] = src[srcY][srcX]
				}
			}
		}
	}

	return circle
}

// ExtractEllipse extracts an elliptical region from a matrix.
//
// Parameters:
//   - src: Source matrix
//   - centerX, centerY: Center of ellipse (in matrix coordinates)
//   - a, b: Semi-major and semi-minor axes (in grid cells)
//   - angle: Rotation angle of ellipse (radians, 0 = horizontal)
//   - fillValue: Value to use for cells outside ellipse (default: 0.0)
//
// Returns extracted ellipse matrix, or nil if bounds invalid.
func ExtractEllipse(src mat.Matrix, centerX, centerY int, a, b, angle, fillValue float32) mat.Matrix {
	rows := len(src)
	cols := len(src[0])

	if a <= 0 || b <= 0 {
		return nil
	}

	// Calculate bounding box
	maxRadius := math32.Max(a, b)
	size := int(2*maxRadius + 1)

	// Check bounds
	if centerX-int(maxRadius) < 0 || centerX+int(maxRadius) >= cols ||
		centerY-int(maxRadius) < 0 || centerY+int(maxRadius) >= rows {
		return nil
	}

	ellipse := mat.New(size, size)

	// Initialize with fill value
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			ellipse[i][j] = fillValue
		}
	}

	// Pre-compute rotation
	cosAngle := math32.Cos(angle)
	sinAngle := math32.Sin(angle)

	// Extract elliptical region
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			// Calculate position relative to center
			dx := float32(j) - maxRadius
			dy := float32(i) - maxRadius

			// Rotate point back to ellipse coordinate system
			rotX := dx*cosAngle + dy*sinAngle
			rotY := -dx*sinAngle + dy*cosAngle

			// Check if point is inside ellipse: (x/a)^2 + (y/b)^2 <= 1
			ellipseValue := (rotX*rotX)/(a*a) + (rotY*rotY)/(b*b)

			if ellipseValue <= 1.0 {
				srcY := centerY - int(maxRadius) + i
				srcX := centerX - int(maxRadius) + j
				if srcY >= 0 && srcY < rows && srcX >= 0 && srcX < cols {
					ellipse[i][j] = src[srcY][srcX]
				}
			}
		}
	}

	return ellipse
}
