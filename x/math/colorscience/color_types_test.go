package colorscience

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewXYZ(t *testing.T) {
	xyz := NewXYZ(0.5, 0.6, 0.7)
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	assert.Equal(t, float32(0.5), X)
	assert.Equal(t, float32(0.6), Y)
	assert.Equal(t, float32(0.7), Z)
}

func TestNewLAB(t *testing.T) {
	lab := NewLAB(50.0, 10.0, -20.0)
	L, a, b := lab.LAB()
	assert.Equal(t, float32(50.0), L)
	assert.Equal(t, float32(10.0), a)
	assert.Equal(t, float32(-20.0), b)
}

func TestNewRGB(t *testing.T) {
	rgb := NewRGB(0.8, 0.9, 1.0)
	R, G, B := rgb.RGB()
	assert.Equal(t, float32(0.8), R)
	assert.Equal(t, float32(0.9), G)
	assert.Equal(t, float32(1.0), B)
}

func TestLABComponents(t *testing.T) {
	lab := NewLAB(75.5, -10.5, 25.3)
	assert.Equal(t, float32(75.5), lab.L())
	assert.Equal(t, float32(-10.5), lab.A())
	assert.Equal(t, float32(25.3), lab.B())
}

func TestRGBComponents(t *testing.T) {
	rgb := NewRGB(128.0, 200.0, 255.0)
	assert.Equal(t, float32(128.0), rgb.R())
	assert.Equal(t, float32(200.0), rgb.G())
	assert.Equal(t, float32(255.0), rgb.B())
}

func TestXYZLuminance(t *testing.T) {
	xyz := NewXYZ(0.5, 0.6, 0.7)
	assert.Equal(t, float32(0.6), xyz.Luminance())
}

func TestXYZToLABMethod(t *testing.T) {
	xyz := NewXYZ(0.5, 0.6, 0.7)
	lab := xyz.ToLAB(WhitePointD65_10)
	
	// Verify LAB values are reasonable (L should be positive, typically 0-100)
	L, a, b := lab.LAB()
	assert.Greater(t, L, float32(0))
	assert.Less(t, L, float32(100))
	// a and b can be negative or positive
	assert.NotNil(t, a)
	assert.NotNil(t, b)
}

func TestXYZToRGBMethod(t *testing.T) {
	xyz := NewXYZ(0.5, 0.6, 0.7)
	
	// Test with out255=false (0-1 range)
	rgb := xyz.ToRGB(false)
	R, G, B := rgb.RGB()
	assert.GreaterOrEqual(t, R, float32(0))
	assert.LessOrEqual(t, R, float32(1))
	assert.GreaterOrEqual(t, G, float32(0))
	assert.LessOrEqual(t, G, float32(1))
	assert.GreaterOrEqual(t, B, float32(0))
	assert.LessOrEqual(t, B, float32(1))
	
	// Test with out255=true (0-255 range)
	rgb255 := xyz.ToRGB(true)
	R255, G255, B255 := rgb255.RGB()
	assert.GreaterOrEqual(t, R255, float32(0))
	assert.LessOrEqual(t, R255, float32(255))
	assert.GreaterOrEqual(t, G255, float32(0))
	assert.LessOrEqual(t, G255, float32(255))
	assert.GreaterOrEqual(t, B255, float32(0))
	assert.LessOrEqual(t, B255, float32(255))
}

func TestLABToXYZMethod(t *testing.T) {
	lab := NewLAB(50.0, 10.0, -20.0)
	xyz := lab.ToXYZ(WhitePointD65_10)
	
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	// XYZ values should be positive
	assert.Greater(t, X, float32(0))
	assert.Greater(t, Y, float32(0))
	assert.Greater(t, Z, float32(0))
}

func TestLABToRGBMethod(t *testing.T) {
	lab := NewLAB(50.0, 10.0, -20.0)
	
	// Test with out255=false
	rgb := lab.ToRGB(WhitePointD65_10, false)
	R, G, B := rgb.RGB()
	assert.GreaterOrEqual(t, R, float32(0))
	assert.LessOrEqual(t, R, float32(1))
	assert.GreaterOrEqual(t, G, float32(0))
	assert.LessOrEqual(t, G, float32(1))
	assert.GreaterOrEqual(t, B, float32(0))
	assert.LessOrEqual(t, B, float32(1))
	
	// Test with out255=true
	rgb255 := lab.ToRGB(WhitePointD65_10, true)
	R255, G255, B255 := rgb255.RGB()
	assert.GreaterOrEqual(t, R255, float32(0))
	assert.LessOrEqual(t, R255, float32(255))
	assert.GreaterOrEqual(t, G255, float32(0))
	assert.LessOrEqual(t, G255, float32(255))
	assert.GreaterOrEqual(t, B255, float32(0))
	assert.LessOrEqual(t, B255, float32(255))
}

func TestRGBToXYZMethod(t *testing.T) {
	// Test with 0-1 range RGB
	rgb := NewRGB(0.5, 0.6, 0.7)
	xyz := rgb.ToXYZ()
	
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	// XYZ values should be positive
	assert.Greater(t, X, float32(0))
	assert.Greater(t, Y, float32(0))
	assert.Greater(t, Z, float32(0))
}

func TestRGBToLAB(t *testing.T) {
	rgb := NewRGB(0.5, 0.6, 0.7)
	lab := rgb.ToLAB(WhitePointD65_10)
	
	L, a, b := lab.LAB()
	assert.Greater(t, L, float32(0))
	assert.Less(t, L, float32(100))
	assert.NotNil(t, a)
	assert.NotNil(t, b)
}

func TestXYZAdapt(t *testing.T) {
	xyz := NewXYZ(0.5, 0.6, 0.7)
	
	// Adapt from D65 to D50
	adapted, err := xyz.Adapt(WhitePointD65_10, WhitePointD50_10, AdaptationBradford)
	require.NoError(t, err)
	
	X, Y, Z := vec.Vector3D(adapted).XYZ()
	// Adapted XYZ should still be positive
	assert.Greater(t, X, float32(0))
	assert.Greater(t, Y, float32(0))
	assert.Greater(t, Z, float32(0))
}

func TestDeltaE76(t *testing.T) {
	// Test with identical colors (should be 0)
	lab1 := NewLAB(50.0, 10.0, -20.0)
	lab2 := NewLAB(50.0, 10.0, -20.0)
	deltaE := DeltaE76(lab1, lab2)
	assert.Equal(t, float32(0), deltaE)
	
	// Test with different colors
	lab3 := NewLAB(50.0, 10.0, -20.0)
	lab4 := NewLAB(60.0, 15.0, -15.0)
	deltaE2 := DeltaE76(lab3, lab4)
	assert.Greater(t, deltaE2, float32(0))
	
	// Verify it's the Euclidean distance
	expected := float32(math.Sqrt(
		math.Pow(float64(50.0-60.0), 2) +
			math.Pow(float64(10.0-15.0), 2) +
			math.Pow(float64(-20.0-(-15.0)), 2),
	))
	assert.InDelta(t, expected, deltaE2, 0.001)
}

func TestDeltaE94(t *testing.T) {
	lab1 := NewLAB(50.0, 10.0, -20.0)
	lab2 := NewLAB(50.0, 10.0, -20.0)
	
	// Identical colors should have delta E close to 0
	deltaE := DeltaE94(lab1, lab2, "graphic")
	assert.InDelta(t, 0, deltaE, 0.1)
	
	// Different colors should have positive delta E
	lab3 := NewLAB(50.0, 10.0, -20.0)
	lab4 := NewLAB(60.0, 15.0, -15.0)
	deltaE2 := DeltaE94(lab3, lab4, "graphic")
	assert.Greater(t, deltaE2, float32(0))
	
	// Test with textiles application type
	deltaE3 := DeltaE94(lab3, lab4, "textiles")
	assert.Greater(t, deltaE3, float32(0))
	// Textiles should give different result than graphic
	assert.NotEqual(t, deltaE2, deltaE3)
}

func TestDeltaE2000(t *testing.T) {
	lab1 := NewLAB(50.0, 10.0, -20.0)
	lab2 := NewLAB(50.0, 10.0, -20.0)
	
	// Identical colors should have delta E close to 0
	deltaE := DeltaE2000(lab1, lab2)
	assert.InDelta(t, 0, deltaE, 0.1)
	
	// Different colors should have positive delta E
	lab3 := NewLAB(50.0, 10.0, -20.0)
	lab4 := NewLAB(60.0, 15.0, -15.0)
	deltaE2 := DeltaE2000(lab3, lab4)
	assert.Greater(t, deltaE2, float32(0))
}

func TestColorConversionRoundTrip(t *testing.T) {
	// Test XYZ -> LAB -> XYZ round trip
	originalXYZ := NewXYZ(0.5, 0.6, 0.7)
	lab := originalXYZ.ToLAB(WhitePointD65_10)
	roundTripXYZ := lab.ToXYZ(WhitePointD65_10)
	
	X1, Y1, Z1 := vec.Vector3D(originalXYZ).XYZ()
	X2, Y2, Z2 := vec.Vector3D(roundTripXYZ).XYZ()
	
	// Should be close (within floating point precision)
	assert.InDelta(t, X1, X2, 0.01)
	assert.InDelta(t, Y1, Y2, 0.01)
	assert.InDelta(t, Z1, Z2, 0.01)
}

func TestColorConversionRoundTripRGB(t *testing.T) {
	// Test RGB -> XYZ -> RGB round trip (0-1 range)
	originalRGB := NewRGB(0.5, 0.6, 0.7)
	xyz := originalRGB.ToXYZ()
	roundTripRGB := xyz.ToRGB(false)
	
	R1, G1, B1 := originalRGB.RGB()
	R2, G2, B2 := roundTripRGB.RGB()
	
	// Should be close (within floating point precision)
	assert.InDelta(t, R1, R2, 0.1)
	assert.InDelta(t, G1, G2, 0.1)
	assert.InDelta(t, B1, B2, 0.1)
}

func TestDeltaE76WithKnownValues(t *testing.T) {
	// Test with known reference values
	// LAB(50, 0, 0) to LAB(60, 0, 0) should have delta E = 10
	lab1 := NewLAB(50.0, 0.0, 0.0)
	lab2 := NewLAB(60.0, 0.0, 0.0)
	deltaE := DeltaE76(lab1, lab2)
	assert.InDelta(t, 10.0, deltaE, 0.001)
	
	// LAB(50, 0, 0) to LAB(50, 10, 0) should have delta E = 10
	lab3 := NewLAB(50.0, 0.0, 0.0)
	lab4 := NewLAB(50.0, 10.0, 0.0)
	deltaE2 := DeltaE76(lab3, lab4)
	assert.InDelta(t, 10.0, deltaE2, 0.001)
}

func TestXYZComponentAccess(t *testing.T) {
	xyz := NewXYZ(0.5, 0.6, 0.7)
	
	// Test direct component access via Vector3D
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	assert.Equal(t, float32(0.5), X)
	assert.Equal(t, float32(0.6), Y)
	assert.Equal(t, float32(0.7), Z)
	
	// Test array indexing
	assert.Equal(t, float32(0.5), xyz[0])
	assert.Equal(t, float32(0.6), xyz[1])
	assert.Equal(t, float32(0.7), xyz[2])
}

func TestLABComponentAccess(t *testing.T) {
	lab := NewLAB(75.5, -10.5, 25.3)
	
	// Test method access
	assert.Equal(t, float32(75.5), lab.L())
	assert.Equal(t, float32(-10.5), lab.A())
	assert.Equal(t, float32(25.3), lab.B())
	
	// Test tuple access
	L, a, b := lab.LAB()
	assert.Equal(t, float32(75.5), L)
	assert.Equal(t, float32(-10.5), a)
	assert.Equal(t, float32(25.3), b)
}

func TestRGBComponentAccess(t *testing.T) {
	rgb := NewRGB(128.0, 200.0, 255.0)
	
	// Test method access
	assert.Equal(t, float32(128.0), rgb.R())
	assert.Equal(t, float32(200.0), rgb.G())
	assert.Equal(t, float32(255.0), rgb.B())
	
	// Test tuple access
	R, G, B := rgb.RGB()
	assert.Equal(t, float32(128.0), R)
	assert.Equal(t, float32(200.0), G)
	assert.Equal(t, float32(255.0), B)
}

