package colorscience

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	// Test with defaults
	cs, err := New()
	require.NoError(t, err)
	assert.NotNil(t, cs)
	assert.Equal(t, Observer10Deg, cs.observer)
	assert.Equal(t, WhitePointD65_10, cs.whitePoint)
	assert.NotNil(t, cs.cmf)
	assert.NotNil(t, cs.illuminant)
}

func TestNewWithOptions(t *testing.T) {
	cs, err := New(
		WithObserver(Observer2Deg),
		WithIlluminant("D50"),
		WithWhitePoint(WhitePointD50_10),
	)
	require.NoError(t, err)
	assert.Equal(t, Observer2Deg, cs.observer)
	assert.Equal(t, WhitePointD50_10, cs.whitePoint)
	assert.NotNil(t, cs.cmf)
	assert.NotNil(t, cs.illuminant)
}

func TestNewWithCustomIlluminant(t *testing.T) {
	customWl := vec.NewFrom(400.0, 450.0, 500.0, 550.0, 600.0)
	customVals := vec.NewFrom(1.0, 1.1, 1.2, 1.1, 1.0)
	customSPD := NewSPD(customWl, customVals)

	cs, err := New(WithIlluminantSPD(customSPD.Matrix))
	require.NoError(t, err)
	assert.NotNil(t, cs)
	assert.Equal(t, customSPD.Matrix, cs.illuminant.Matrix)
}

func TestLoadCMF(t *testing.T) {
	cmf, err := LoadCMF(Observer10Deg)
	require.NoError(t, err)
	assert.NotNil(t, cmf)
	assert.Greater(t, cmf.Len(), 0)

	wlSPD := cmf.Wavelengths()
	xBarSPD := cmf.XBar()
	yBarSPD := cmf.YBar()
	zBarSPD := cmf.ZBar()

	assert.Equal(t, cmf.Len(), wlSPD.Len())
	assert.Equal(t, cmf.Len(), xBarSPD.Len())
	assert.Equal(t, cmf.Len(), yBarSPD.Len())
	assert.Equal(t, cmf.Len(), zBarSPD.Len())

	// Also test Values methods
	assert.Equal(t, cmf.Len(), cmf.WavelengthsValues().Len())
	assert.Equal(t, cmf.Len(), cmf.XBarValues().Len())
	assert.Equal(t, cmf.Len(), cmf.YBarValues().Len())
	assert.Equal(t, cmf.Len(), cmf.ZBarValues().Len())

	cmf2, err := LoadCMF(Observer2Deg)
	require.NoError(t, err)
	assert.NotNil(t, cmf2)
}

func TestLoadIlluminantSPD(t *testing.T) {
	illuminant, err := LoadIlluminantSPD("D65")
	require.NoError(t, err)
	assert.NotNil(t, illuminant)
	assert.Greater(t, illuminant.Len(), 0)
	assert.Equal(t, illuminant.Len(), illuminant.Wavelengths().Len())
	assert.Equal(t, illuminant.Len(), illuminant.Values().Len())

	illuminant2, err := LoadIlluminantSPD("D50")
	require.NoError(t, err)
	assert.NotNil(t, illuminant2)

	illuminant3, err := LoadIlluminantSPD("A")
	require.NoError(t, err)
	assert.NotNil(t, illuminant3)
}

func TestLoadIlluminantSPDInvalid(t *testing.T) {
	_, err := LoadIlluminantSPD("INVALID")
	assert.Error(t, err)
}

func TestAvailableIlluminants(t *testing.T) {
	illuminants := AvailableIlluminants()
	assert.Greater(t, len(illuminants), 0)

	// Check that expected illuminants are present
	illuminantMap := make(map[string]bool)
	for _, name := range illuminants {
		illuminantMap[name] = true
	}

	assert.True(t, illuminantMap["D65"], "D65 should be available")
	assert.True(t, illuminantMap["D50"], "D50 should be available")
	assert.True(t, illuminantMap["A"], "A should be available")
}

func TestComputeXYZ(t *testing.T) {
	cs, err := New()
	require.NoError(t, err)

	// Input SPD with lower resolution (measured spectrum)
	wavelengths := vec.NewFrom(400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0)
	spdValues := vec.NewFrom(0.5, 0.6, 0.7, 0.65, 0.55, 0.5, 0.45)

	// Create 2-row matrix: row 0 = wavelengths, row 1 = values
	spdMatrix := NewSPD(wavelengths, spdValues)

	xyz, err := cs.ComputeXYZ(spdMatrix.Matrix)
	require.NoError(t, err)
	require.False(t, isNaN(xyz[0]) || isNaN(xyz[1]) || isNaN(xyz[2]), "XYZ values should not be NaN")

	// XYZ values should be positive
	assert.True(t, xyz[0] > 0, "X should be positive, got %f", xyz[0])
	assert.True(t, xyz[1] > 0, "Y should be positive, got %f", xyz[1])
	assert.True(t, xyz[2] > 0, "Z should be positive, got %f", xyz[2])
}

func isNaN(f float32) bool {
	return f != f
}

func TestComputeXYZMismatchedLengths(t *testing.T) {
	cs, err := New()
	require.NoError(t, err)

	wavelengths := vec.NewFrom(400.0, 450.0, 500.0)
	spdValues := vec.NewFrom(0.5, 0.6) // Different length

	// Create 2-row matrix: row 0 = wavelengths, row 1 = values
	spdMatrix := NewSPD(wavelengths, spdValues)

	_, err = cs.ComputeXYZ(spdMatrix.Matrix)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "lengths must match")
}

func TestCalibrate(t *testing.T) {
	// Create SPD with initial wavelengths (these will be replaced by calibration)
	initialWl := vec.NewFrom(0.0, 0.0, 0.0, 0.0, 0.0) // Unknown wavelengths
	values := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)
	spd := NewSPD(initialWl, values)

	// Calibrate with pairs: (index, wavelength)
	// Index 0 = 400nm, index 2 = 500nm, index 4 = 600nm
	calibrated, err := spd.Calibrate(0, 400.0, 2, 500.0, 4, 600.0)
	require.NoError(t, err)
	assert.NotNil(t, calibrated)
	assert.Equal(t, 5, calibrated.Len())

	// Check calibrated wavelengths
	calWl := calibrated.Wavelengths()
	// Calibration points should be exact
	assert.Equal(t, float32(400.0), calWl[0]) // Exact calibration point
	assert.Equal(t, float32(500.0), calWl[2]) // Exact calibration point
	assert.Equal(t, float32(600.0), calWl[4]) // Exact calibration point

	// Interpolated values should be between calibration points (cubic spline gives smooth curve)
	assert.True(t, calWl[1] > 400.0 && calWl[1] < 500.0, "index 1 should be between 400 and 500")
	assert.True(t, calWl[3] > 500.0 && calWl[3] < 600.0, "index 3 should be between 500 and 600")

	// Check monotonicity (wavelengths should be strictly increasing)
	assert.True(t, calWl[0] < calWl[1], "wavelengths should be increasing")
	assert.True(t, calWl[1] < calWl[2], "wavelengths should be increasing")
	assert.True(t, calWl[2] < calWl[3], "wavelengths should be increasing")
	assert.True(t, calWl[3] < calWl[4], "wavelengths should be increasing")

	// Check that values remain unchanged
	calibratedVals := calibrated.Values()
	assert.Equal(t, float32(1.0), calibratedVals[0])
	assert.Equal(t, float32(2.0), calibratedVals[1])
	assert.Equal(t, float32(3.0), calibratedVals[2])
	assert.Equal(t, float32(4.0), calibratedVals[3])
	assert.Equal(t, float32(5.0), calibratedVals[4])
}

func TestCalibrateTwoPoints(t *testing.T) {
	// Test with just two calibration points
	initialWl := vec.NewFrom(0.0, 0.0, 0.0, 0.0, 0.0)
	values := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)
	spd := NewSPD(initialWl, values)

	// Calibrate: index 0 = 400nm, index 4 = 700nm
	calibrated, err := spd.Calibrate(0, 400.0, 4, 700.0)
	require.NoError(t, err)

	calWl := calibrated.Wavelengths()
	// Calibration points should be exact
	assert.Equal(t, float32(400.0), calWl[0])
	assert.Equal(t, float32(700.0), calWl[4])

	// With only 2 points, Catmull-Rom uses cubic interpolation
	// Values should be between the calibration points
	assert.True(t, calWl[1] > 400.0 && calWl[1] < 700.0, "index 1 should be between 400 and 700")
	assert.True(t, calWl[2] > 400.0 && calWl[2] < 700.0, "index 2 should be between 400 and 700")
	assert.True(t, calWl[3] > 400.0 && calWl[3] < 700.0, "index 3 should be between 400 and 700")

	// Check monotonicity (wavelengths should be strictly increasing)
	assert.True(t, calWl[0] < calWl[1] && calWl[1] < calWl[2] && calWl[2] < calWl[3] && calWl[3] < calWl[4])
}

func TestCalibrateOutOfRange(t *testing.T) {
	initialWl := vec.NewFrom(0.0, 0.0, 0.0)
	values := vec.NewFrom(1.0, 2.0, 3.0)
	spd := NewSPD(initialWl, values)

	// Try to calibrate with out-of-range index
	_, err := spd.Calibrate(0, 400.0, 10, 700.0) // Index 10 is out of range
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "out of range")
}

func TestCalibrateNilInput(t *testing.T) {
	var spd SPD
	_, err := spd.Calibrate(0, 400.0)
	assert.Error(t, err)
}

func TestCalibrateEmptyMapping(t *testing.T) {
	initialWl := vec.NewFrom(0.0, 0.0, 0.0)
	values := vec.NewFrom(1.0, 2.0, 3.0)
	spd := NewSPD(initialWl, values)

	_, err := spd.Calibrate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be empty")
}

func TestCalibrateOddPairs(t *testing.T) {
	initialWl := vec.NewFrom(0.0, 0.0, 0.0)
	values := vec.NewFrom(1.0, 2.0, 3.0)
	spd := NewSPD(initialWl, values)

	_, err := spd.Calibrate(0, 400.0, 1.0) // Odd number of arguments
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be even")
}

func TestCalibrateSinglePoint(t *testing.T) {
	initialWl := vec.NewFrom(0.0, 0.0, 0.0)
	values := vec.NewFrom(1.0, 2.0, 3.0)
	spd := NewSPD(initialWl, values)

	_, err := spd.Calibrate(0, 400.0) // Only one point
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "at least 2 calibration points")
}

func TestCalibrateDuplicateIndices(t *testing.T) {
	initialWl := vec.NewFrom(0.0, 0.0, 0.0)
	values := vec.NewFrom(1.0, 2.0, 3.0)
	spd := NewSPD(initialWl, values)

	_, err := spd.Calibrate(0, 400.0, 0, 450.0) // Duplicate index
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")
}

func TestXYZToLAB(t *testing.T) {
	// Test with D65 white point - when XYZ equals white point, LAB should be (100, 0, 0)
	X, Y, Z := float32(94.81), float32(100.0), float32(107.32) // D65_10 white point
	L, a, b := XYZToLAB(X, Y, Z, WhitePointD65_10)

	// At white point, L should be close to 100, a and b close to 0
	assert.InDelta(t, 100.0, L, 0.1)
	assert.InDelta(t, 0.0, a, 0.1)
	assert.InDelta(t, 0.0, b, 0.1)
}

func TestLABToXYZ(t *testing.T) {
	// Test round-trip conversion
	X1, Y1, Z1 := float32(50.0), float32(60.0), float32(70.0)
	L, a, b := XYZToLAB(X1, Y1, Z1, WhitePointD65_10)
	X2, Y2, Z2 := LABToXYZ(L, a, b, WhitePointD65_10)

	assert.InDelta(t, X1, X2, 0.1)
	assert.InDelta(t, Y1, Y2, 0.1)
	assert.InDelta(t, Z1, Z2, 0.1)
}

func TestXYZToRGB(t *testing.T) {
	// Test D65 white point -> should give white RGB
	X, Y, Z := float32(95.047), float32(100.0), float32(108.883)
	r, g, b := XYZToRGB(X, Y, Z, false)

	// White should map to approximately (1, 1, 1)
	assert.InDelta(t, 1.0, r, 0.1)
	assert.InDelta(t, 1.0, g, 0.1)
	assert.InDelta(t, 1.0, b, 0.1)
}

func TestRGBToXYZ(t *testing.T) {
	// Test round-trip conversion
	r1, g1, b1 := float32(0.5), float32(0.6), float32(0.7)
	X, Y, Z := RGBToXYZ(r1, g1, b1)
	r2, g2, b2 := XYZToRGB(X, Y, Z, false)

	assert.InDelta(t, r1, r2, 0.1)
	assert.InDelta(t, g1, g2, 0.1)
	assert.InDelta(t, b1, b2, 0.1)
}

func TestRGBToXYZ255(t *testing.T) {
	// Test with 0-255 range
	r, g, b := float32(128), float32(128), float32(128)
	X, Y, Z := RGBToXYZ(r, g, b)

	// Should produce valid XYZ values
	assert.True(t, X > 0)
	assert.True(t, Y > 0)
	assert.True(t, Z > 0)
}

func TestAdaptXYZ(t *testing.T) {
	X, Y, Z := float32(50.0), float32(60.0), float32(70.0)

	Xa, Ya, Za, err := AdaptXYZ(X, Y, Z, WhitePointD50_10, WhitePointD65_10, AdaptationBradford)
	require.NoError(t, err)

	// Adapted values should be different but reasonable
	assert.NotEqual(t, X, Xa)
	assert.True(t, Xa > 0)
	assert.True(t, Ya > 0)
	assert.True(t, Za > 0)
}

func TestAdaptXYZInvalidMethod(t *testing.T) {
	_, _, _, err := AdaptXYZ(50, 60, 70, WhitePointD50_10, WhitePointD65_10, AdaptationMethod("invalid"))
	assert.Error(t, err)
}

func TestIntegrationKernel(t *testing.T) {
	wavelengths := vec.NewFrom(400.0, 410.0, 420.0, 430.0, 440.0)
	kernel := IntegrationKernel(wavelengths)

	assert.Equal(t, wavelengths.Len(), kernel.Len())

	// First and last should be half intervals
	assert.InDelta(t, 5.0, kernel[0], 0.1) // (410-400)/2
	assert.InDelta(t, 5.0, kernel[4], 0.1) // (440-430)/2

	// Middle should be average of adjacent intervals
	assert.InDelta(t, 10.0, kernel[2], 0.1) // (420-410 + 430-420)/2
}

func TestIntegrate(t *testing.T) {
	wavelengths := vec.NewFrom(0.0, 1.0, 2.0, 3.0, 4.0)
	values := vec.NewFrom(1.0, 1.0, 1.0, 1.0, 1.0) // Constant function

	result := Integrate(wavelengths, values)

	// Integral of constant 1 from 0 to 4 should be 4
	assert.InDelta(t, 4.0, result, 0.1)
}

func TestSPDMatrix(t *testing.T) {
	wavelengths := vec.NewFrom(400.0, 450.0, 500.0)
	values := vec.NewFrom(1.0, 2.0, 3.0)

	spd := NewSPD(wavelengths, values)
	assert.NotNil(t, spd)
	assert.Equal(t, 3, spd.Len())
	assert.Equal(t, wavelengths, spd.Wavelengths())
	assert.Equal(t, values, spd.Values())
}

func TestObserverCMFMatrix(t *testing.T) {
	cmf, err := LoadCMF(Observer10Deg)
	require.NoError(t, err)

	assert.Equal(t, 4, cmf.Rows())
	assert.Greater(t, cmf.Len(), 0)

	wlSPD := cmf.Wavelengths()
	xBarSPD := cmf.XBar()
	yBarSPD := cmf.YBar()
	zBarSPD := cmf.ZBar()

	assert.Equal(t, cmf.Len(), wlSPD.Len())
	assert.Equal(t, cmf.Len(), xBarSPD.Len())
	assert.Equal(t, cmf.Len(), yBarSPD.Len())
	assert.Equal(t, cmf.Len(), zBarSPD.Len())

	// Test Values methods
	assert.Equal(t, cmf.Len(), cmf.WavelengthsValues().Len())
	assert.Equal(t, cmf.Len(), cmf.XBarValues().Len())
	assert.Equal(t, cmf.Len(), cmf.YBarValues().Len())
	assert.Equal(t, cmf.Len(), cmf.ZBarValues().Len())
}

func TestSPDInterpolate(t *testing.T) {
	// Create SPD with lower resolution
	sourceWl := vec.NewFrom(400.0, 500.0, 600.0, 700.0)
	sourceVals := vec.NewFrom(1.0, 2.0, 3.0, 4.0)
	spd := NewSPD(sourceWl, sourceVals)

	// Interpolate to higher resolution (like CMF/illuminant)
	targetWl := vec.NewFrom(400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0)
	interpSPD := spd.Interpolate(targetWl)

	assert.Equal(t, targetWl.Len(), interpSPD.Len())
	assert.Equal(t, targetWl, interpSPD.Wavelengths())

	// Check interpolation: value at 450 should be between 1.0 and 2.0
	interpVals := interpSPD.Values()
	assert.Greater(t, interpVals[1], float32(1.0))
	assert.Less(t, interpVals[1], float32(2.0))
}
