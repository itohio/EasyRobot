package grid

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Line draws a line between two points in the matrix using Bresenham's algorithm
// dst: Destination matrix to draw into
// x0, y0: Start point (column, row)
// x1, y1: End point (column, row)
// value: Value to set for line pixels
// Returns dst (modified in-place)
func Line(dst mat.Matrix, x0, y0, x1, y1 int, value float32) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	rows := len(dst)
	cols := len(dst[0])

	dx := x1 - x0
	if dx < 0 {
		dx = -dx
	}
	dy := y1 - y0
	if dy < 0 {
		dy = -dy
	}

	sx := 1
	if x0 > x1 {
		sx = -1
	}
	sy := 1
	if y0 > y1 {
		sy = -1
	}

	err := dx - dy

	x, y := x0, y0

	for {
		if x >= 0 && x < cols && y >= 0 && y < rows {
			dst[y][x] = value
		}

		if x == x1 && y == y1 {
			break
		}

		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}

	return dst
}

// Path draws a path (sequence of connected lines) into the matrix
// dst: Destination matrix to draw into
// path: Sequence of 2D vectors representing path points
// value: Value to set for path pixels
// Returns dst (modified in-place)
func Path(dst mat.Matrix, path []vec.Vector2D, value float32) mat.Matrix {
	if dst == nil || len(path) < 2 {
		return dst
	}

	for i := 0; i < len(path)-1; i++ {
		x0 := int(path[i][0])
		y0 := int(path[i][1])
		x1 := int(path[i+1][0])
		y1 := int(path[i+1][1])
		Line(dst, x0, y0, x1, y1, value)
	}

	return dst
}

// HLine draws an optimized horizontal line
// dst: Destination matrix to draw into
// x0, x1: Start and end x coordinates (columns)
// y: Row coordinate (same for entire line)
// value: Value to set for line pixels
// Returns dst (modified in-place)
func HLine(dst mat.Matrix, x0, x1, y int, value float32) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	if y < 0 || y >= len(dst) {
		return dst
	}

	cols := len(dst[0])

	if x0 > x1 {
		x0, x1 = x1, x0
	}

	if x0 < 0 {
		x0 = 0
	}
	if x1 >= cols {
		x1 = cols - 1
	}

	row := dst[y]
	for x := x0; x <= x1; x++ {
		if x >= 0 && x < cols {
			row[x] = value
		}
	}

	return dst
}

// VLine draws an optimized vertical line
// dst: Destination matrix to draw into
// x: Column coordinate (same for entire line)
// y0, y1: Start and end y coordinates (rows)
// value: Value to set for line pixels
// Returns dst (modified in-place)
func VLine(dst mat.Matrix, x, y0, y1 int, value float32) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	if x < 0 || x >= len(dst[0]) {
		return dst
	}

	rows := len(dst)

	if y0 > y1 {
		y0, y1 = y1, y0
	}

	if y0 < 0 {
		y0 = 0
	}
	if y1 >= rows {
		y1 = rows - 1
	}

	for y := y0; y <= y1; y++ {
		if y >= 0 && y < rows {
			dst[y][x] = value
		}
	}

	return dst
}

// GradientLine draws a line with gradient values between start and end
// dst: Destination matrix to draw into
// x0, y0: Start point (column, row)
// x1, y1: End point (column, row)
// value0: Value at start point
// value1: Value at end point
// Returns dst (modified in-place)
func GradientLine(dst mat.Matrix, x0, y0, x1, y1 int, value0, value1 float32) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	dx := x1 - x0
	if dx < 0 {
		dx = -dx
	}
	dy := y1 - y0
	if dy < 0 {
		dy = -dy
	}

	sx := 1
	if x0 > x1 {
		sx = -1
	}
	sy := 1
	if y0 > y1 {
		sy = -1
	}

	err := dx - dy

	x, y := x0, y0
	totalDist := math32.Sqrt(float32((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)))
	t := float32(0.0)

	for {
		if x >= 0 && x < len(dst[0]) && y >= 0 && y < len(dst) {
			dist := math32.Sqrt(float32((x-x0)*(x-x0) + (y-y0)*(y-y0)))
			if totalDist > 0 {
				t = dist / totalDist
			} else {
				t = 0.0
			}
			value := value0 + t*(value1-value0)
			dst[y][x] = value
		}

		if x == x1 && y == y1 {
			break
		}

		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}

	return dst
}

// Rectangle draws a rectangle into the matrix
// dst: Destination matrix to draw into
// x, y: Top-left corner (column, row)
// width, height: Width and height of rectangle
// value: Value to set for rectangle pixels
// filled: If true, fills rectangle; if false, only draws outline
// Returns dst (modified in-place)
func Rectangle(dst mat.Matrix, x, y, width, height int, value float32, filled bool) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	rows := len(dst)
	cols := len(dst[0])

	if filled {
		for i := y; i < y+height; i++ {
			if i < 0 || i >= rows {
				continue
			}
			for j := x; j < x+width; j++ {
				if j >= 0 && j < cols {
					dst[i][j] = value
				}
			}
		}
	} else {
		// Top edge
		Line(dst, x, y, x+width-1, y, value)
		// Bottom edge
		Line(dst, x, y+height-1, x+width-1, y+height-1, value)
		// Left edge
		Line(dst, x, y, x, y+height-1, value)
		// Right edge
		Line(dst, x+width-1, y, x+width-1, y+height-1, value)
	}

	return dst
}

// Circle draws a circle into the matrix using midpoint circle algorithm
// dst: Destination matrix to draw into
// centerX, centerY: Center of circle (column, row)
// radius: Radius of circle (in grid cells)
// value: Value to set for circle pixels
// filled: If true, fills circle; if false, only draws outline
// Returns dst (modified in-place)
func Circle(dst mat.Matrix, centerX, centerY, radius int, value float32, filled bool) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 || radius <= 0 {
		return dst
	}

	rows := len(dst)
	cols := len(dst[0])

	if filled {
		radiusSq := float32(radius) * float32(radius)
		for i := centerY - radius; i <= centerY+radius; i++ {
			if i < 0 || i >= rows {
				continue
			}
			for j := centerX - radius; j <= centerX+radius; j++ {
				if j < 0 || j >= cols {
					continue
				}
				dx := float32(j - centerX)
				dy := float32(i - centerY)
				distSq := dx*dx + dy*dy
				if distSq <= radiusSq {
					dst[i][j] = value
				}
			}
		}
	} else {
		// Midpoint circle algorithm for outline
		x := 0
		y := radius
		d := 1 - radius

		setPixel := func(x, y int) {
			cx, cy := centerX, centerY
			if cx+x >= 0 && cx+x < cols && cy+y >= 0 && cy+y < rows {
				dst[cy+y][cx+x] = value
			}
			if cx-x >= 0 && cx-x < cols && cy+y >= 0 && cy+y < rows {
				dst[cy+y][cx-x] = value
			}
			if cx+x >= 0 && cx+x < cols && cy-y >= 0 && cy-y < rows {
				dst[cy-y][cx+x] = value
			}
			if cx-x >= 0 && cx-x < cols && cy-y >= 0 && cy-y < rows {
				dst[cy-y][cx-x] = value
			}
			if cx+y >= 0 && cx+y < cols && cy+x >= 0 && cy+x < rows {
				dst[cy+x][cx+y] = value
			}
			if cx-y >= 0 && cx-y < cols && cy+x >= 0 && cy+x < rows {
				dst[cy+x][cx-y] = value
			}
			if cx+y >= 0 && cx+y < cols && cy-x >= 0 && cy-x < rows {
				dst[cy-x][cx+y] = value
			}
			if cx-y >= 0 && cx-y < cols && cy-x >= 0 && cy-x < rows {
				dst[cy-x][cx-y] = value
			}
		}

		for x <= y {
			setPixel(x, y)

			if d < 0 {
				d += 2*x + 3
			} else {
				d += 2*(x-y) + 5
				y--
			}
			x++
		}
	}

	return dst
}

// Ellipse draws an ellipse into the matrix
// dst: Destination matrix to draw into
// centerX, centerY: Center of ellipse (column, row)
// a, b: Semi-major and semi-minor axes (in grid cells)
// angle: Rotation angle of ellipse (radians, 0 = horizontal)
// value: Value to set for ellipse pixels
// filled: If true, fills ellipse; if false, only draws outline
// Returns dst (modified in-place)
func Ellipse(dst mat.Matrix, centerX, centerY int, a, b, angle float32, value float32, filled bool) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 || a <= 0 || b <= 0 {
		return dst
	}

	rows := len(dst)
	cols := len(dst[0])

	cosAngle := math32.Cos(angle)
	sinAngle := math32.Sin(angle)

	maxRadius := math32.Max(a, b)
	radiusInt := int(maxRadius) + 1

	if filled {
		for i := centerY - radiusInt; i <= centerY+radiusInt; i++ {
			if i < 0 || i >= rows {
				continue
			}
			for j := centerX - radiusInt; j <= centerX+radiusInt; j++ {
				if j < 0 || j >= cols {
					continue
				}

				dx := float32(j - centerX)
				dy := float32(i - centerY)

				rotX := dx*cosAngle + dy*sinAngle
				rotY := -dx*sinAngle + dy*cosAngle

				ellipseValue := (rotX*rotX)/(a*a) + (rotY*rotY)/(b*b)

				if ellipseValue <= 1.0 {
					dst[i][j] = value
				}
			}
		}
	} else {
		// Approximate outline using parametric equation
		steps := int(2 * math32.Pi * maxRadius)
		if steps < 16 {
			steps = 16
		}
		stepSize := 2 * math32.Pi / float32(steps)

		var prevX, prevY int
		first := true

		for t := float32(0); t <= 2*math32.Pi; t += stepSize {
			x := a * math32.Cos(t)
			y := b * math32.Sin(t)

			rotX := x*cosAngle - y*sinAngle
			rotY := x*sinAngle + y*cosAngle

			px := int(rotX) + centerX
			py := int(rotY) + centerY

			if px >= 0 && px < cols && py >= 0 && py < rows {
				if !first {
					Line(dst, prevX, prevY, px, py, value)
				}
				prevX, prevY = px, py
				first = false
			}
		}
	}

	return dst
}

// Triangle draws a triangle outline into the matrix
// dst: Destination matrix to draw into
// x0, y0: First vertex (column, row)
// x1, y1: Second vertex (column, row)
// x2, y2: Third vertex (column, row)
// value: Value to set for triangle edges
// Returns dst (modified in-place)
func Triangle(dst mat.Matrix, x0, y0, x1, y1, x2, y2 int, value float32) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	// Draw three edges of the triangle
	Line(dst, x0, y0, x1, y1, value)
	Line(dst, x1, y1, x2, y2, value)
	Line(dst, x2, y2, x0, y0, value)

	return dst
}

// ShadedTriangle draws a filled triangle with interpolated shading
// dst: Destination matrix to draw into
// x0, y0: First vertex (column, row)
// x1, y1: Second vertex (column, row)
// x2, y2: Third vertex (column, row)
// value0: Value at first vertex
// value1: Value at second vertex
// value2: Value at third vertex
// Returns dst (modified in-place)
func ShadedTriangle(dst mat.Matrix, x0, y0, x1, y1, x2, y2 int, value0, value1, value2 float32) mat.Matrix {
	if dst == nil || len(dst) == 0 || len(dst[0]) == 0 {
		return dst
	}

	rows := len(dst)
	cols := len(dst[0])

	// Sort vertices by y coordinate (top to bottom)
	vertices := []struct {
		x, y int
		v    float32
	}{
		{x0, y0, value0},
		{x1, y1, value1},
		{x2, y2, value2},
	}

	// Sort by y coordinate
	for i := 0; i < 2; i++ {
		for j := i + 1; j < 3; j++ {
			if vertices[i].y > vertices[j].y {
				vertices[i], vertices[j] = vertices[j], vertices[i]
			}
		}
	}

	v0 := vertices[0]
	v1 := vertices[1]
	v2 := vertices[2]

	// Handle degenerate case: all vertices are the same
	if v0.x == v1.x && v0.y == v1.y && v1.x == v2.x && v1.y == v2.y {
		if v0.x >= 0 && v0.x < cols && v0.y >= 0 && v0.y < rows {
			dst[v0.y][v0.x] = v0.v
		}
		return dst
	}

	// Calculate bounding box
	minY := v0.y
	maxY := v2.y
	if minY < 0 {
		minY = 0
	}
	if maxY >= rows {
		maxY = rows - 1
	}

	// For each scanline from top to bottom
	for y := minY; y <= maxY; y++ {
		if y < 0 || y >= rows {
			continue
		}

		var xLeft, xRight int
		var vLeft, vRight float32

		// Find x coordinates and values for this scanline
		if y < v1.y {
			// Between v0 and v1
			t := float32(0)
			if v1.y != v0.y {
				t = float32(y-v0.y) / float32(v1.y-v0.y)
			}
			xLeft = int(float32(v0.x) + t*float32(v1.x-v0.x))
			vLeft = v0.v + t*(v1.v-v0.v)

			// Between v0 and v2
			t = float32(0)
			if v2.y != v0.y {
				t = float32(y-v0.y) / float32(v2.y-v0.y)
			}
			xRight = int(float32(v0.x) + t*float32(v2.x-v0.x))
			vRight = v0.v + t*(v2.v-v0.v)
		} else {
			// Between v1 and v2
			t := float32(0)
			if v2.y != v1.y {
				t = float32(y-v1.y) / float32(v2.y-v1.y)
			}
			xLeft = int(float32(v1.x) + t*float32(v2.x-v1.x))
			vLeft = v1.v + t*(v2.v-v1.v)

			// Between v0 and v2
			t = float32(0)
			if v2.y != v0.y {
				t = float32(y-v0.y) / float32(v2.y-v0.y)
			}
			xRight = int(float32(v0.x) + t*float32(v2.x-v0.x))
			vRight = v0.v + t*(v2.v-v0.v)
		}

		// Ensure left < right
		if xLeft > xRight {
			xLeft, xRight = xRight, xLeft
			vLeft, vRight = vRight, vLeft
		}

		// Clamp to bounds
		if xLeft < 0 {
			xLeft = 0
		}
		if xRight >= cols {
			xRight = cols - 1
		}

		// Draw horizontal line with gradient
		if xRight >= xLeft {
			width := xRight - xLeft + 1
			for x := xLeft; x <= xRight; x++ {
				if x >= 0 && x < cols {
					t := float32(0)
					if width > 1 {
						t = float32(x-xLeft) / float32(width-1)
					}
					value := vLeft + t*(vRight-vLeft)
					dst[y][x] = value
				}
			}
		}
	}

	return dst
}
