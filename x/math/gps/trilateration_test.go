package gps

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestTrilateration(t *testing.T) {
	// Test trilateration with three known points
	// Use simple test case: points forming a right triangle
	P1 := ECEF(vec.Vector3D{0, 0, 0})
	P2 := ECEF(vec.Vector3D{1000, 0, 0})
	P3 := ECEF(vec.Vector3D{0, 1000, 0})

	// Unknown point at (500, 500, 0)
	expectedX := float32(500)
	expectedY := float32(500)
	expectedZ := float32(0)

	// Calculate actual distances
	DistA := math32.Sqrt(expectedX*expectedX + expectedY*expectedY + expectedZ*expectedZ)
	DistB := math32.Sqrt((expectedX-1000)*(expectedX-1000) + expectedY*expectedY + expectedZ*expectedZ)
	DistC := math32.Sqrt(expectedX*expectedX + (expectedY-1000)*(expectedY-1000) + expectedZ*expectedZ)

	result := Trilateration(P1, P2, P3, DistA, DistB, DistC)

	// Verify result matches expected
	tolerance := float32(1.0) // 1m tolerance
	assert.InDelta(t, float64(expectedX), float64(result.X()), float64(tolerance), "X should match")
	assert.InDelta(t, float64(expectedY), float64(result.Y()), float64(tolerance), "Y should match")
	assert.InDelta(t, float64(expectedZ), float64(result.Z()), float64(tolerance), "Z should match")

	// Verify distances match
	var resultVec vec.Vector3D
	resultVec = vec.Vector3D(result)
	distToP1 := vec.Vector3D(P1).Sub(resultVec).(vec.Vector3D).Magnitude()
	distToP2 := vec.Vector3D(P2).Sub(resultVec).(vec.Vector3D).Magnitude()
	distToP3 := vec.Vector3D(P3).Sub(resultVec).(vec.Vector3D).Magnitude()

	assert.InDelta(t, float64(DistA), float64(distToP1), float64(tolerance), "Distance to P1 should match")
	assert.InDelta(t, float64(DistB), float64(distToP2), float64(tolerance), "Distance to P2 should match")
	assert.InDelta(t, float64(DistC), float64(distToP3), float64(tolerance), "Distance to P3 should match")
}

func TestTrilateration_3D(t *testing.T) {
	// Test trilateration in 3D space
	P1 := ECEF(vec.Vector3D{0, 0, 0})
	P2 := ECEF(vec.Vector3D{1000, 0, 0})
	P3 := ECEF(vec.Vector3D{0, 0, 1000})

	// Unknown point at (500, 0, 500)
	expectedX := float32(500)
	expectedY := float32(0)
	expectedZ := float32(500)

	DistA := math32.Sqrt(expectedX*expectedX + expectedY*expectedY + expectedZ*expectedZ)
	DistB := math32.Sqrt((expectedX-1000)*(expectedX-1000) + expectedY*expectedY + expectedZ*expectedZ)
	DistC := math32.Sqrt(expectedX*expectedX + expectedY*expectedY + (expectedZ-1000)*(expectedZ-1000))

	result := Trilateration(P1, P2, P3, DistA, DistB, DistC)

	tolerance := float32(1.0)
	assert.InDelta(t, float64(expectedX), float64(result.X()), float64(tolerance), "X should match")
	assert.InDelta(t, float64(expectedY), float64(result.Y()), float64(tolerance), "Y should match")
	assert.InDelta(t, float64(expectedZ), float64(result.Z()), float64(tolerance), "Z should match")

	// Verify distances
	var resultVec vec.Vector3D
	resultVec = vec.Vector3D(result)
	distToP1 := vec.Vector3D(P1).Sub(resultVec).(vec.Vector3D).Magnitude()
	distToP2 := vec.Vector3D(P2).Sub(resultVec).(vec.Vector3D).Magnitude()
	distToP3 := vec.Vector3D(P3).Sub(resultVec).(vec.Vector3D).Magnitude()

	assert.InDelta(t, float64(DistA), float64(distToP1), float64(tolerance), "Distance to P1 should match")
	assert.InDelta(t, float64(DistB), float64(distToP2), float64(tolerance), "Distance to P2 should match")
	assert.InDelta(t, float64(DistC), float64(distToP3), float64(tolerance), "Distance to P3 should match")
}

func TestTrilateration_InvalidPoints(t *testing.T) {
	// Test with identical points (should return zero ECEF)
	P1 := ECEF(vec.Vector3D{100, 200, 300})
	P2 := ECEF(vec.Vector3D{100, 200, 300}) // Same as P1
	P3 := ECEF(vec.Vector3D{0, 0, 0})

	result := Trilateration(P1, P2, P3, 100, 100, 100)
	assert.Equal(t, ECEF{}, result, "Should return zero ECEF for invalid points")

	// Test with collinear points
	P1 = ECEF(vec.Vector3D{0, 0, 0})
	P2 = ECEF(vec.Vector3D{100, 0, 0})
	P3 = ECEF(vec.Vector3D{200, 0, 0}) // Collinear with P1 and P2

	result = Trilateration(P1, P2, P3, 50, 50, 50)
	assert.Equal(t, ECEF{}, result, "Should return zero ECEF for collinear points")
}

func TestTrilateration_EquilateralTriangle(t *testing.T) {
	// Test with equilateral triangle configuration
	P1 := ECEF(vec.Vector3D{0, 0, 0})
	P2 := ECEF(vec.Vector3D{1000, 0, 0})
	P3 := ECEF(vec.Vector3D{500, 866.0254, 0}) // Equilateral triangle (height = 1000*sin(60°))

	// Unknown point at center of triangle (centroid)
	expectedX := float32(500)
	expectedY := float32(288.675) // 866.0254 / 3
	expectedZ := float32(0)

	DistA := math32.Sqrt(expectedX*expectedX + expectedY*expectedY + expectedZ*expectedZ)
	DistB := math32.Sqrt((expectedX-1000)*(expectedX-1000) + expectedY*expectedY + expectedZ*expectedZ)
	DistC := math32.Sqrt((expectedX-500)*(expectedX-500) + (expectedY-866.0254)*(expectedY-866.0254) + expectedZ*expectedZ)

	result := Trilateration(P1, P2, P3, DistA, DistB, DistC)

	tolerance := float32(1.0)
	assert.InDelta(t, float64(expectedX), float64(result.X()), float64(tolerance), "X should match")
	assert.InDelta(t, float64(expectedY), float64(result.Y()), float64(tolerance), "Y should match")
	assert.InDelta(t, float64(expectedZ), float64(result.Z()), float64(tolerance), "Z should match")
}

func TestTrilateration_GeneralCase(t *testing.T) {
	// Test with general non-symmetric configuration
	P1 := ECEF(vec.Vector3D{100, 200, 300})
	P2 := ECEF(vec.Vector3D{1500, 400, 500})
	P3 := ECEF(vec.Vector3D{200, 1800, 100})

	// Unknown point at arbitrary location
	expectedX := float32(600)
	expectedY := float32(800)
	expectedZ := float32(400)

	// Calculate distances
	var expectedVec, p1Vec, p2Vec, p3Vec vec.Vector3D
	expectedVec = vec.Vector3D{expectedX, expectedY, expectedZ}
	p1Vec = vec.Vector3D(P1)
	p2Vec = vec.Vector3D(P2)
	p3Vec = vec.Vector3D(P3)

	DistA := p1Vec.Sub(expectedVec).(vec.Vector3D).Magnitude()
	DistB := p2Vec.Sub(expectedVec).(vec.Vector3D).Magnitude()
	DistC := p3Vec.Sub(expectedVec).(vec.Vector3D).Magnitude()

	result := Trilateration(P1, P2, P3, DistA, DistB, DistC)

	tolerance := float32(1.0)
	assert.InDelta(t, float64(expectedX), float64(result.X()), float64(tolerance), "X should match")
	assert.InDelta(t, float64(expectedY), float64(result.Y()), float64(tolerance), "Y should match")
	assert.InDelta(t, float64(expectedZ), float64(result.Z()), float64(tolerance), "Z should match")

	// Verify distances
	var resultVec vec.Vector3D
	resultVec = vec.Vector3D(result)
	distToP1 := p1Vec.Sub(resultVec).(vec.Vector3D).Magnitude()
	distToP2 := p2Vec.Sub(resultVec).(vec.Vector3D).Magnitude()
	distToP3 := p3Vec.Sub(resultVec).(vec.Vector3D).Magnitude()

	assert.InDelta(t, float64(DistA), float64(distToP1), float64(tolerance), "Distance to P1 should match")
	assert.InDelta(t, float64(DistB), float64(distToP2), float64(tolerance), "Distance to P2 should match")
	assert.InDelta(t, float64(DistC), float64(distToP3), float64(tolerance), "Distance to P3 should match")
}

func TestTrilateration_TableDriven(t *testing.T) {
	tests := []struct {
		name      string
		P1        ECEF
		P2        ECEF
		P3        ECEF
		expected  ECEF
		tolerance float32
	}{
		{
			name:      "Origin centered",
			P1:        ECEF(vec.Vector3D{-500, 0, 0}),
			P2:        ECEF(vec.Vector3D{500, 0, 0}),
			P3:        ECEF(vec.Vector3D{0, 500, 0}),
			expected:  ECEF(vec.Vector3D{0, 0, 0}),
			tolerance: 1.0,
		},
		{
			name:      "Offset from origin",
			P1:        ECEF(vec.Vector3D{1000, 1000, 0}),
			P2:        ECEF(vec.Vector3D{2000, 1000, 0}),
			P3:        ECEF(vec.Vector3D{1500, 2000, 0}),
			expected:  ECEF(vec.Vector3D{1500, 1500, 0}),
			tolerance: 1.0,
		},
		{
			name:      "3D offset",
			P1:        ECEF(vec.Vector3D{0, 0, 0}),
			P2:        ECEF(vec.Vector3D{200, 0, 0}),
			P3:        ECEF(vec.Vector3D{0, 200, 0}),
			expected:  ECEF(vec.Vector3D{100, 100, 100}),
			tolerance: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate distances from expected point
			var expectedVec, p1Vec, p2Vec, p3Vec vec.Vector3D
			expectedVec = vec.Vector3D(tt.expected)
			p1Vec = vec.Vector3D(tt.P1)
			p2Vec = vec.Vector3D(tt.P2)
			p3Vec = vec.Vector3D(tt.P3)

			DistA := p1Vec.Sub(expectedVec).(vec.Vector3D).Magnitude()
			DistB := p2Vec.Sub(expectedVec).(vec.Vector3D).Magnitude()
			DistC := p3Vec.Sub(expectedVec).(vec.Vector3D).Magnitude()

			result := Trilateration(tt.P1, tt.P2, tt.P3, DistA, DistB, DistC)

			assert.InDelta(t, float64(tt.expected.X()), float64(result.X()), float64(tt.tolerance), "X coordinate")
			assert.InDelta(t, float64(tt.expected.Y()), float64(result.Y()), float64(tt.tolerance), "Y coordinate")
			assert.InDelta(t, float64(tt.expected.Z()), float64(result.Z()), float64(tt.tolerance), "Z coordinate")
		})
	}
}

func TestMultilateration_3Points(t *testing.T) {
	// Test multilateration with 3 points (should use trilateration)
	points := []struct {
		Coord Coord
		ECEF  ECEF
	}{
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{1000, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 1000, 0})},
	}

	// Unknown point at (500, 500, 0)
	distances := []float32{
		math32.Sqrt(500*500 + 500*500),
		math32.Sqrt(500*500 + 500*500),
		math32.Sqrt(500*500 + 500*500),
	}

	result, err := Multilateration(points, distances)
	assert.NoError(t, err)

	tolerance := float32(10.0)
	assert.InDelta(t, 500.0, float64(result.X()), float64(tolerance), "X should be approximately 500")
	assert.InDelta(t, 500.0, float64(result.Y()), float64(tolerance), "Y should be approximately 500")
	assert.InDelta(t, 0.0, float64(result.Z()), float64(tolerance), "Z should be approximately 0")
}

func TestMultilateration_4Points(t *testing.T) {
	// Test multilateration with 4 points (should use least squares)
	points := []struct {
		Coord Coord
		ECEF  ECEF
	}{
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{1000, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 1000, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 0, 1000})},
	}

	// Unknown point at (500, 500, 500)
	expectedX := float32(500)
	expectedY := float32(500)
	expectedZ := float32(500)

	distances := []float32{
		math32.Sqrt(expectedX*expectedX + expectedY*expectedY + expectedZ*expectedZ),
		math32.Sqrt((expectedX-1000)*(expectedX-1000) + expectedY*expectedY + expectedZ*expectedZ),
		math32.Sqrt(expectedX*expectedX + (expectedY-1000)*(expectedY-1000) + expectedZ*expectedZ),
		math32.Sqrt(expectedX*expectedX + expectedY*expectedY + (expectedZ-1000)*(expectedZ-1000)),
	}

	result, err := Multilateration(points, distances)
	assert.NoError(t, err)

	tolerance := float32(50.0) // Larger tolerance for least squares
	assert.InDelta(t, float64(expectedX), float64(result.X()), float64(tolerance), "X should be approximately 500")
	assert.InDelta(t, float64(expectedY), float64(result.Y()), float64(tolerance), "Y should be approximately 500")
	assert.InDelta(t, float64(expectedZ), float64(result.Z()), float64(tolerance), "Z should be approximately 500")
}

func TestMultilateration_Errors(t *testing.T) {
	// Test insufficient points
	points := []struct {
		Coord Coord
		ECEF  ECEF
	}{
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{1000, 0, 0})},
	}
	distances := []float32{100, 200}

	result, err := Multilateration(points, distances)
	assert.Error(t, err)
	assert.Equal(t, ErrInsufficientPoints, err)
	assert.Equal(t, ECEF{}, result)

	// Test mismatched lengths
	points = []struct {
		Coord Coord
		ECEF  ECEF
	}{
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{1000, 0, 0})},
		{Coord(vec.Vector3D{0, 0, 0}), ECEF(vec.Vector3D{0, 1000, 0})},
	}
	distances = []float32{100, 200} // Only 2 distances for 3 points

	result, err = Multilateration(points, distances)
	assert.Error(t, err)
	assert.Equal(t, ErrMismatchedLengths, err)
	assert.Equal(t, ECEF{}, result)
}

