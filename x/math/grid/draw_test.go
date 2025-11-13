package grid

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	"github.com/stretchr/testify/assert"
)

func TestLine_Horizontal(t *testing.T) {
	dst := mat.New(10, 10)
	Line(dst, 0, 5, 9, 5, 1.0)

	// Check all points on line are set
	for x := 0; x < 10; x++ {
		assert.Equal(t, float32(1.0), dst[5][x], "Horizontal line point at x=%d", x)
	}
}

func TestLine_Vertical(t *testing.T) {
	dst := mat.New(10, 10)
	Line(dst, 5, 0, 5, 9, 1.0)

	// Check all points on line are set
	for y := 0; y < 10; y++ {
		assert.Equal(t, float32(1.0), dst[y][5], "Vertical line point at y=%d", y)
	}
}

func TestLine_Diagonal(t *testing.T) {
	dst := mat.New(10, 10)
	Line(dst, 0, 0, 9, 9, 1.0)

	// Check diagonal points are set
	for i := 0; i < 10; i++ {
		assert.Equal(t, float32(1.0), dst[i][i], "Diagonal line point at (%d,%d)", i, i)
	}
}

func TestLine_OutOfBounds(t *testing.T) {
	dst := mat.New(5, 5)
	Line(dst, -5, -5, 10, 10, 1.0)

	// Should not crash, only valid points set
	assert.NotNil(t, dst, "Should not crash with out-of-bounds coordinates")
}

func TestPath_Simple(t *testing.T) {
	dst := mat.New(10, 10)
	path := []vec.Vector2D{
		{0, 0},
		{5, 5},
		{9, 9},
	}

	Path(dst, path, 1.0)

	// Check start and end points
	assert.Equal(t, float32(1.0), dst[0][0], "Path start point")
	assert.Equal(t, float32(1.0), dst[9][9], "Path end point")
}

func TestPath_Empty(t *testing.T) {
	dst := mat.New(10, 10)
	path := []vec.Vector2D{}

	Path(dst, path, 1.0)
	// Should not crash
	assert.NotNil(t, dst)
}

func TestPath_SinglePoint(t *testing.T) {
	dst := mat.New(10, 10)
	path := []vec.Vector2D{
		{5, 5},
	}

	Path(dst, path, 1.0)
	// With only one point, no line is drawn, so value should remain 0
	// This is expected behavior (need at least 2 points to draw)
}

func TestRectangle_Outline(t *testing.T) {
	dst := mat.New(10, 10)
	Rectangle(dst, 2, 2, 5, 5, 1.0, false)

	// Check top edge
	for x := 2; x < 7; x++ {
		assert.Equal(t, float32(1.0), dst[2][x], "Top edge at x=%d", x)
	}
	// Check bottom edge
	for x := 2; x < 7; x++ {
		assert.Equal(t, float32(1.0), dst[6][x], "Bottom edge at x=%d", x)
	}
	// Check left edge
	for y := 2; y < 7; y++ {
		assert.Equal(t, float32(1.0), dst[y][2], "Left edge at y=%d", y)
	}
	// Check right edge
	for y := 2; y < 7; y++ {
		assert.Equal(t, float32(1.0), dst[y][6], "Right edge at y=%d", y)
	}

	// Check interior is not filled
	assert.Equal(t, float32(0.0), dst[3][3], "Interior should not be filled")
}

func TestRectangle_Filled(t *testing.T) {
	dst := mat.New(10, 10)
	Rectangle(dst, 2, 2, 5, 5, 1.0, true)

	// Check all points are filled
	for y := 2; y < 7; y++ {
		for x := 2; x < 7; x++ {
			assert.Equal(t, float32(1.0), dst[y][x], "Filled rectangle at (%d,%d)", x, y)
		}
	}
}

func TestCircle_Outline(t *testing.T) {
	dst := mat.New(20, 20)
	Circle(dst, 10, 10, 5, 1.0, false)

	// Check some expected points on circle outline
	// Center point should not be set (outline only)
	assert.Equal(t, float32(0.0), dst[10][10], "Center should not be set for outline")

	// Points on circle should be set
	// Using approximate check - circle algorithm should set points near radius
	count := 0
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			if dst[y][x] == 1.0 {
				count++
			}
		}
	}
	assert.Greater(t, count, 0, "Circle outline should have some points set")
}

func TestCircle_Filled(t *testing.T) {
	dst := mat.New(20, 20)
	Circle(dst, 10, 10, 5, 1.0, true)

	// Check center is set
	assert.Equal(t, float32(1.0), dst[10][10], "Center should be set for filled circle")

	// Check points inside circle are set
	assert.Equal(t, float32(1.0), dst[10][10], "Point at center")
	assert.Equal(t, float32(1.0), dst[10][12], "Point near edge in x")
	assert.Equal(t, float32(1.0), dst[12][10], "Point near edge in y")
}

func TestCircle_OutOfBounds(t *testing.T) {
	dst := mat.New(5, 5)
	Circle(dst, 10, 10, 5, 1.0, true)

	// Should not crash
	assert.NotNil(t, dst)
}

func TestEllipse_Outline(t *testing.T) {
	dst := mat.New(20, 20)
	Ellipse(dst, 10, 10, 5.0, 3.0, 0.0, 1.0, false)

	// Outline should have points set
	count := 0
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			if dst[y][x] == 1.0 {
				count++
			}
		}
	}
	assert.Greater(t, count, 0, "Ellipse outline should have some points set")
}

func TestEllipse_Filled(t *testing.T) {
	dst := mat.New(20, 20)
	Ellipse(dst, 10, 10, 5.0, 3.0, 0.0, 1.0, true)

	// Check center is set
	assert.Equal(t, float32(1.0), dst[10][10], "Center should be set for filled ellipse")
}

func TestEllipse_Rotated(t *testing.T) {
	dst := mat.New(20, 20)
	// 45 degree rotation
	Ellipse(dst, 10, 10, 5.0, 3.0, 0.7853981633974483, 1.0, true)

	// Should not crash
	assert.NotNil(t, dst)
	assert.Equal(t, float32(1.0), dst[10][10], "Center should be set")
}

func TestLine_NilMatrix(t *testing.T) {
	result := Line(nil, 0, 0, 10, 10, 1.0)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestPath_NilMatrix(t *testing.T) {
	path := []vec.Vector2D{{0, 0}, {10, 10}}
	result := Path(nil, path, 1.0)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestRectangle_NilMatrix(t *testing.T) {
	result := Rectangle(nil, 0, 0, 10, 10, 1.0, false)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestCircle_NilMatrix(t *testing.T) {
	result := Circle(nil, 10, 10, 5, 1.0, false)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestEllipse_NilMatrix(t *testing.T) {
	result := Ellipse(nil, 10, 10, 5.0, 3.0, 0.0, 1.0, false)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestHLine_Simple(t *testing.T) {
	dst := mat.New(10, 10)
	HLine(dst, 2, 7, 5, 1.0)

	// Check all points on line are set
	for x := 2; x <= 7; x++ {
		assert.Equal(t, float32(1.0), dst[5][x], "Horizontal line point at x=%d", x)
	}
	// Check points outside line are not set
	assert.Equal(t, float32(0.0), dst[5][1], "Point before line should not be set")
	assert.Equal(t, float32(0.0), dst[5][8], "Point after line should not be set")
}

func TestHLine_Reversed(t *testing.T) {
	dst := mat.New(10, 10)
	HLine(dst, 7, 2, 5, 1.0)

	// Check all points on line are set (should handle reversed coordinates)
	for x := 2; x <= 7; x++ {
		assert.Equal(t, float32(1.0), dst[5][x], "Horizontal line point at x=%d", x)
	}
}

func TestHLine_OutOfBounds(t *testing.T) {
	dst := mat.New(10, 10)
	HLine(dst, -5, 15, 5, 1.0)

	// Should clamp to bounds
	for x := 0; x < 10; x++ {
		assert.Equal(t, float32(1.0), dst[5][x], "Horizontal line should be clamped at x=%d", x)
	}
}

func TestHLine_InvalidY(t *testing.T) {
	dst := mat.New(10, 10)
	result := HLine(dst, 2, 7, -1, 1.0)
	assert.Equal(t, dst, result, "Should return dst unchanged for invalid y")
}

func TestVLine_Simple(t *testing.T) {
	dst := mat.New(10, 10)
	VLine(dst, 5, 2, 7, 1.0)

	// Check all points on line are set
	for y := 2; y <= 7; y++ {
		assert.Equal(t, float32(1.0), dst[y][5], "Vertical line point at y=%d", y)
	}
	// Check points outside line are not set
	assert.Equal(t, float32(0.0), dst[1][5], "Point before line should not be set")
	assert.Equal(t, float32(0.0), dst[8][5], "Point after line should not be set")
}

func TestVLine_Reversed(t *testing.T) {
	dst := mat.New(10, 10)
	VLine(dst, 5, 7, 2, 1.0)

	// Check all points on line are set (should handle reversed coordinates)
	for y := 2; y <= 7; y++ {
		assert.Equal(t, float32(1.0), dst[y][5], "Vertical line point at y=%d", y)
	}
}

func TestVLine_OutOfBounds(t *testing.T) {
	dst := mat.New(10, 10)
	VLine(dst, 5, -5, 15, 1.0)

	// Should clamp to bounds
	for y := 0; y < 10; y++ {
		assert.Equal(t, float32(1.0), dst[y][5], "Vertical line should be clamped at y=%d", y)
	}
}

func TestVLine_InvalidX(t *testing.T) {
	dst := mat.New(10, 10)
	result := VLine(dst, -1, 2, 7, 1.0)
	assert.Equal(t, dst, result, "Should return dst unchanged for invalid x")
}

func TestGradientLine_Horizontal(t *testing.T) {
	dst := mat.New(10, 10)
	GradientLine(dst, 0, 5, 9, 5, 0.0, 1.0)

	// Check start point
	assert.Equal(t, float32(0.0), dst[5][0], "Gradient line start value")
	// Check end point
	assert.Equal(t, float32(1.0), dst[5][9], "Gradient line end value")
	// Check middle point
	assert.Greater(t, dst[5][4], float32(0.0), "Gradient line middle should be > start")
	assert.Less(t, dst[5][4], float32(1.0), "Gradient line middle should be < end")
}

func TestGradientLine_Vertical(t *testing.T) {
	dst := mat.New(10, 10)
	GradientLine(dst, 5, 0, 5, 9, 0.0, 1.0)

	// Check start point
	assert.Equal(t, float32(0.0), dst[0][5], "Gradient line start value")
	// Check end point
	assert.Equal(t, float32(1.0), dst[9][5], "Gradient line end value")
}

func TestGradientLine_Diagonal(t *testing.T) {
	dst := mat.New(10, 10)
	GradientLine(dst, 0, 0, 9, 9, 0.0, 1.0)

	// Check start point
	assert.Equal(t, float32(0.0), dst[0][0], "Gradient line start value")
	// Check end point
	assert.Equal(t, float32(1.0), dst[9][9], "Gradient line end value")
}

func TestGradientLine_SamePoint(t *testing.T) {
	dst := mat.New(10, 10)
	GradientLine(dst, 5, 5, 5, 5, 0.5, 0.5)

	// Should set the single point
	assert.Equal(t, float32(0.5), dst[5][5], "Gradient line at same point")
}

func TestGradientLine_NilMatrix(t *testing.T) {
	result := GradientLine(nil, 0, 0, 10, 10, 0.0, 1.0)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestTriangle_Simple(t *testing.T) {
	dst := mat.New(10, 10)
	Triangle(dst, 2, 2, 7, 2, 4, 8, 1.0)

	// Check vertices are set
	assert.Equal(t, float32(1.0), dst[2][2], "Triangle vertex 1")
	assert.Equal(t, float32(1.0), dst[2][7], "Triangle vertex 2")
	assert.Equal(t, float32(1.0), dst[8][4], "Triangle vertex 3")
}

func TestTriangle_Degenerate(t *testing.T) {
	dst := mat.New(10, 10)
	Triangle(dst, 5, 5, 5, 5, 5, 5, 1.0)

	// All vertices same - should set single point three times
	assert.Equal(t, float32(1.0), dst[5][5], "Triangle at same point")
}

func TestTriangle_NilMatrix(t *testing.T) {
	result := Triangle(nil, 0, 0, 5, 5, 10, 10, 1.0)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestShadedTriangle_Simple(t *testing.T) {
	dst := mat.New(10, 10)
	ShadedTriangle(dst, 2, 2, 7, 2, 4, 8, 0.0, 0.5, 1.0)

	// Check vertices have correct values
	assert.Equal(t, float32(0.0), dst[2][2], "Shaded triangle vertex 1")
	assert.Equal(t, float32(0.5), dst[2][7], "Shaded triangle vertex 2")
	assert.Equal(t, float32(1.0), dst[8][4], "Shaded triangle vertex 3")
}

func TestShadedTriangle_Interpolated(t *testing.T) {
	dst := mat.New(20, 20)
	// Large triangle with gradient
	ShadedTriangle(dst, 5, 5, 15, 5, 10, 15, 0.0, 0.5, 1.0)

	// Check top edge has interpolated values
	assert.Equal(t, float32(0.0), dst[5][5], "Shaded triangle top-left")
	assert.Equal(t, float32(0.5), dst[5][15], "Shaded triangle top-right")

	// Check interior has interpolated values
	midY := 10
	midX := 10
	assert.Greater(t, dst[midY][midX], float32(0.0), "Shaded triangle interior should be interpolated")
	assert.Less(t, dst[midY][midX], float32(1.0), "Shaded triangle interior should be interpolated")
}

func TestShadedTriangle_Degenerate(t *testing.T) {
	dst := mat.New(10, 10)
	ShadedTriangle(dst, 5, 5, 5, 5, 5, 5, 0.0, 0.5, 1.0)

	// All vertices same - should set single point to first value
	assert.Equal(t, float32(0.0), dst[5][5], "Shaded triangle at same point")
}

func TestShadedTriangle_NilMatrix(t *testing.T) {
	result := ShadedTriangle(nil, 0, 0, 5, 5, 10, 10, 0.0, 0.5, 1.0)
	assert.Nil(t, result, "Should return nil for nil matrix")
}

func TestShadedTriangle_OutOfBounds(t *testing.T) {
	dst := mat.New(10, 10)
	// Triangle partially out of bounds
	ShadedTriangle(dst, -5, -5, 15, 15, 10, 20, 0.0, 0.5, 1.0)

	// Should not crash and should fill valid region
	assert.NotNil(t, dst, "Should not crash with out-of-bounds triangle")
}
