package colorscience

import (
	"fmt"

	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Luminance calculates luminance (cd/m²) from CIE XYZ tristimulus values.
// Luminance is given by the Y component of XYZ.
// For absolute luminance, Y should be in cd/m² (requires calibration).
// For relative luminance, Y is normalized (0-100).
//
// Parameters:
//   - X, Y, Z: CIE XYZ tristimulus values
//
// Returns: luminance (Y component in cd/m² or normalized 0-100)
func Luminance(X, Y, Z float32) float32 {
	return Y
}

// ComputeLuminance computes luminance from a spectral power distribution matrix.
// Uses the configured CMF and illuminant from ColorScience.
// Returns absolute or relative luminance depending on the SPD units.
//
// Parameters:
//   - cs: ColorScience instance with configured CMF and illuminant
//   - spdMatrix: Spectral power distribution matrix (1 row: values only, or 2 rows: wavelengths + values)
//
// Returns: luminance (Y component), error
func (cs *ColorScience) ComputeLuminance(spdMatrix matTypes.Matrix) (float32, error) {
	xyz, err := cs.ComputeXYZ(spdMatrix)
	if err != nil {
		return 0, fmt.Errorf("failed to calculate luminance from SPD: %w", err)
	}
	return xyz.Luminance(), nil
}

// CalibrateSensorSensitivity calibrates sensor spectral sensitivity using known reference measurements.
// Given a reference SPD with known spectral power and sensor responses to that SPD,
// calculates the sensor's spectral sensitivity (responsivity) function.
//
// The calibration process:
//  1. Measures sensor response to known reference SPD (calibration lamp, standard illuminant)
//  2. Calculates sensitivity = response / reference_SPD at each wavelength
//  3. Returns calibrated sensitivity SPD (wavelengths, sensitivity values)
//
// Parameters:
//   - referenceSPD: Known reference SPD (e.g., standard illuminant or calibration lamp)
//   - sensorResponseSPD: Measured sensor response to the reference SPD (must have same wavelengths)
//
// Returns: calibrated sensitivity SPD (wavelengths, sensitivity values), error
//
// Example:
//   - Reference: D65 illuminant with known spectral power
//   - Sensor response: measured values when exposed to D65
//   - Result: sensor sensitivity function (responsivity at each wavelength)
func CalibrateSensorSensitivity(referenceSPD, sensorResponseSPD SPD) (SPD, error) {
	if referenceSPD.Matrix == nil || sensorResponseSPD.Matrix == nil {
		return SPD{}, fmt.Errorf("reference and sensor response SPDs cannot be nil")
	}

	refVals := referenceSPD.Values()
	refWl := referenceSPD.Wavelengths()
	respVals := sensorResponseSPD.Values()
	respWl := sensorResponseSPD.Wavelengths()

	if refVals == nil || refWl == nil || respVals == nil || respWl == nil {
		return SPD{}, fmt.Errorf("SPDs must have valid wavelengths and values")
	}

	if refWl.Len() != respWl.Len() {
		return SPD{}, fmt.Errorf("reference and sensor response SPDs must have same length")
	}

	// Interpolate to common wavelength grid if needed
	if !wavelengthsMatch(refWl, respWl) {
		refWlVec := vecTypes.Vector(refWl)
		respInterp := sensorResponseSPD.Interpolate(refWlVec)
		respVals = respInterp.Values()
		respWl = refWl
	}

	// Calculate sensitivity = response / reference (element-wise division)
	sensitivity := vec.New(refWl.Len())
	for i := 0; i < sensitivity.Len(); i++ {
		if refVals[i] != 0 {
			sensitivity[i] = respVals[i] / refVals[i]
		} else {
			// Avoid division by zero - set to 0 if reference is zero
			sensitivity[i] = 0
		}
	}

	return NewSPD(refWl, sensitivity), nil
}

// CalibrateSensorSensitivityWithDark calibrates sensor spectral sensitivity accounting for dark current.
// Similar to CalibrateSensorSensitivity but subtracts dark current from sensor response first.
//
// The calibration process:
//  1. Subtracts dark current: corrected_response = response - dark
//  2. Calculates sensitivity = corrected_response / reference_SPD at each wavelength
//  3. Returns calibrated sensitivity SPD
//
// Parameters:
//   - referenceSPD: Known reference SPD (e.g., standard illuminant or calibration lamp)
//   - sensorResponseSPD: Measured sensor response to the reference SPD
//   - darkSPD: Dark current measurement (sensor response with no light)
//
// Returns: calibrated sensitivity SPD, error
func CalibrateSensorSensitivityWithDark(referenceSPD, sensorResponseSPD, darkSPD SPD) (SPD, error) {
	if referenceSPD.Matrix == nil || sensorResponseSPD.Matrix == nil || darkSPD.Matrix == nil {
		return SPD{}, fmt.Errorf("reference, sensor response, and dark SPDs cannot be nil")
	}

	refVals := referenceSPD.Values()
	refWl := referenceSPD.Wavelengths()
	respVals := sensorResponseSPD.Values()
	respWl := sensorResponseSPD.Wavelengths()
	darkVals := darkSPD.Values()
	darkWl := darkSPD.Wavelengths()

	if refVals == nil || refWl == nil || respVals == nil || respWl == nil || darkVals == nil || darkWl == nil {
		return SPD{}, fmt.Errorf("SPDs must have valid wavelengths and values")
	}

	// Interpolate all to common wavelength grid (use reference wavelengths as target)
	refWlVec := vecTypes.Vector(refWl)
	respInterp := sensorResponseSPD.Interpolate(refWlVec)
	darkInterp := darkSPD.Interpolate(refWlVec)

	respVals = respInterp.Values()
	darkVals = darkInterp.Values()

	if respVals.Len() != darkVals.Len() || darkVals.Len() != refVals.Len() {
		return SPD{}, fmt.Errorf("interpolated SPDs must have same length")
	}

	// Calculate sensitivity = (response - dark) / reference
	sensitivity := vec.New(refWl.Len())
	for i := 0; i < sensitivity.Len(); i++ {
		correctedResponse := respVals[i] - darkVals[i]
		if refVals[i] != 0 {
			sensitivity[i] = correctedResponse / refVals[i]
		} else {
			// Avoid division by zero - set to 0 if reference is zero
			sensitivity[i] = 0
		}
	}

	return NewSPD(refWl, sensitivity), nil
}

// wavelengthsMatch checks if two wavelength vectors match (within tolerance).
func wavelengthsMatch(wl1, wl2 vecTypes.Vector) bool {
	if wl1 == nil || wl2 == nil {
		return false
	}
	if wl1.Len() != wl2.Len() {
		return false
	}
	w1 := wl1.View().(vec.Vector)
	w2 := wl2.View().(vec.Vector)
	// Check if wavelengths match within 0.1nm tolerance
	for i := 0; i < w1.Len(); i++ {
		if w1[i]-w2[i] > 0.1 || w2[i]-w1[i] > 0.1 {
			return false
		}
	}
	return true
}
