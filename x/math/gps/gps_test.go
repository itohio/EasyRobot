package gps

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

func TestHaversineDistance(t *testing.T) {
	// Test distance between two known points
	// New York City: 40.7128° N, 74.0060° W
	// Los Angeles: 34.0522° N, 118.2437° W
	// Expected distance: approximately 3944 km
	lat1 := float32(40.7128)
	lon1 := float32(-74.0060)
	lat2 := float32(34.0522)
	lon2 := float32(-118.2437)

	distance := HaversineDistance(lat1, lon1, lat2, lon2)
	expectedDistance := float32(3944000) // meters
	tolerance := float32(50000)          // 50km tolerance

	assert.InDelta(t, float64(expectedDistance), float64(distance), float64(tolerance), "Distance should be approximately 3944 km")
}

func TestCourseAndDistance(t *testing.T) {
	// Test course and distance calculation
	lat1 := float32(40.7128)
	lon1 := float32(-74.0060)
	lat2 := float32(34.0522)
	lon2 := float32(-118.2437)

	result := CourseAndDistanceBetween(lat1, lon1, lat2, lon2)

	// Distance should be positive
	assert.Greater(t, result.Distance(), float32(0), "Distance should be positive")

	// Course should be between 0 and 360 degrees
	assert.GreaterOrEqual(t, result.Course(), float32(0), "Course should be >= 0")
	assert.Less(t, result.Course(), float32(360), "Course should be < 360")
}

func TestToECEF(t *testing.T) {
	// Test conversion of geographic coordinates to ECEF
	// Test with a known point: Equator at 0° longitude
	coord := Coord(vec.Vector3D{0.0, 0.0, 0.0}) // [Lat, Lon, Alt]

	ecef := ToECEF(coord)

	// At equator, 0° longitude, altitude 0:
	// X should be approximately WGS84Axis
	// Y should be approximately 0
	// Z should be approximately 0
	assert.InDelta(t, float64(WGS84Axis), float64(ecef.X()), 1.0, "X should be approximately WGS84Axis")
	assert.InDelta(t, 0.0, float64(ecef.Y()), 1.0, "Y should be approximately 0")
	assert.InDelta(t, 0.0, float64(ecef.Z()), 1.0, "Z should be approximately 0")
}

func TestToECEF_NorthPole(t *testing.T) {
	// Test with North Pole
	coord := Coord(vec.Vector3D{90.0, 0.0, 0.0}) // [Lat, Lon, Alt]

	ecef := ToECEF(coord)

	// At North Pole:
	// X should be approximately 0
	// Y should be approximately 0
	// Z should be approximately WGS84Axis
	assert.InDelta(t, 0.0, float64(ecef.X()), 1.0, "X should be approximately 0")
	assert.InDelta(t, 0.0, float64(ecef.Y()), 1.0, "Y should be approximately 0")
	assert.InDelta(t, float64(WGS84Axis), float64(ecef.Z()), 1.0, "Z should be approximately WGS84Axis")
}

func TestCoord_Methods(t *testing.T) {
	coord := Coord(vec.Vector3D{40.7128, -74.0060, 100.5})

	assert.InDelta(t, 40.7128, float64(coord.Lat()), 0.0001, "Lat should match")
	assert.InDelta(t, -74.0060, float64(coord.Lon()), 0.0001, "Lon should match")
	assert.InDelta(t, 100.5, float64(coord.Alt()), 0.1, "Alt should match")
}

func TestECEF_Methods(t *testing.T) {
	ecef := ECEF(vec.Vector3D{1234.5, 5678.9, 9012.3})

	assert.InDelta(t, 1234.5, float64(ecef.X()), 0.1, "X should match")
	assert.InDelta(t, 5678.9, float64(ecef.Y()), 0.1, "Y should match")
	assert.InDelta(t, 9012.3, float64(ecef.Z()), 0.1, "Z should match")
}

func TestHaversineDistance_ShortDistance(t *testing.T) {
	// Test short distance (same location)
	distance := HaversineDistance(40.7128, -74.0060, 40.7128, -74.0060)
	assert.InDelta(t, 0.0, float64(distance), 1.0, "Distance should be approximately 0 for same location")
}

func TestHaversineDistance_Equator(t *testing.T) {
	// Test distance along equator (1 degree longitude at equator ≈ 111 km)
	distance := HaversineDistance(0.0, 0.0, 0.0, 1.0)
	expectedDistance := float32(111000) // approximately 111 km
	tolerance := float32(1000)          // 1km tolerance
	assert.InDelta(t, float64(expectedDistance), float64(distance), float64(tolerance), "1 degree at equator should be approximately 111 km")
}

func TestCourseAndDistanceBetween_SamePoint(t *testing.T) {
	result := CourseAndDistanceBetween(40.7128, -74.0060, 40.7128, -74.0060)
	assert.InDelta(t, 0.0, float64(result.Distance()), 1.0, "Distance should be 0 for same point")
	// Course is undefined for same point, but should still be in valid range
	assert.GreaterOrEqual(t, result.Course(), float32(0))
	assert.Less(t, result.Course(), float32(360))
}

func TestCourseAndDistanceBetween_North(t *testing.T) {
	// Test course to the north (should be approximately 0 or 360 degrees)
	lat1 := float32(40.0)
	lon1 := float32(-74.0)
	lat2 := float32(41.0) // North
	lon2 := float32(-74.0)

	result := CourseAndDistanceBetween(lat1, lon1, lat2, lon2)
	assert.Greater(t, result.Distance(), float32(0), "Distance should be positive")
	// Course should be close to 0 (north) or 360
	course := result.Course()
	assert.True(t, course < 10 || course > 350, "Course should be approximately north (0 or 360 degrees)")
}

func TestToECEF_RoundTrip(t *testing.T) {
	// Test that ToECEF produces reasonable results for known locations
	tests := []struct {
		name      string
		coord     Coord
		expected  ECEF
		tolerance float32
	}{
		{
			name:      "Equator at Prime Meridian",
			coord:     Coord(vec.Vector3D{0.0, 0.0, 0.0}),
			expected:  ECEF(vec.Vector3D{WGS84Axis, 0.0, 0.0}),
			tolerance: 1.0,
		},
		{
			name:      "North Pole",
			coord:     Coord(vec.Vector3D{90.0, 0.0, 0.0}),
			expected:  ECEF(vec.Vector3D{0.0, 0.0, WGS84Axis}),
			tolerance: 1.0,
		},
		{
			name:      "South Pole",
			coord:     Coord(vec.Vector3D{-90.0, 0.0, 0.0}),
			expected:  ECEF(vec.Vector3D{0.0, 0.0, -WGS84Axis}),
			tolerance: 1.0,
		},
		{
			name:      "90 degrees East",
			coord:     Coord(vec.Vector3D{0.0, 90.0, 0.0}),
			expected:  ECEF(vec.Vector3D{0.0, WGS84Axis, 0.0}),
			tolerance: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ToECEF(tt.coord)
			assert.InDelta(t, float64(tt.expected.X()), float64(result.X()), float64(tt.tolerance), "X coordinate")
			assert.InDelta(t, float64(tt.expected.Y()), float64(result.Y()), float64(tt.tolerance), "Y coordinate")
			assert.InDelta(t, float64(tt.expected.Z()), float64(result.Z()), float64(tt.tolerance), "Z coordinate")
		})
	}
}

func TestToECEF_WithAltitude(t *testing.T) {
	// Test with altitude
	coord := Coord(vec.Vector3D{0.0, 0.0, 1000.0}) // 1km altitude
	ecef := ToECEF(coord)

	// At equator, 0° longitude, with altitude:
	// X should be approximately WGS84Axis + altitude
	// Y and Z should be approximately 0
	assert.InDelta(t, float64(WGS84Axis+1000.0), float64(ecef.X()), 1.0, "X should be WGS84Axis + altitude")
	assert.InDelta(t, 0.0, float64(ecef.Y()), 1.0, "Y should be approximately 0")
	assert.InDelta(t, 0.0, float64(ecef.Z()), 1.0, "Z should be approximately 0")
}
