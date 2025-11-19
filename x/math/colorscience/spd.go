package colorscience

import (
	"encoding/csv"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/math/interpolation"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// SPD represents a Spectral Power Distribution as a 2-row matrix.
// Row 0: wavelengths
// Row 1: values
// SPD embeds matTypes.Matrix interface, allowing methods to be attached while implementing the interface.
type SPD struct {
	matTypes.Matrix
}

// ObserverCMF contains Color Matching Functions as a 4-row matrix.
// Row 0: wavelengths
// Row 1: XBar
// Row 2: YBar
// Row 3: ZBar
// ObserverCMF embeds matTypes.Matrix interface, allowing methods to be attached while implementing the interface.
type ObserverCMF struct {
	matTypes.Matrix
}

// NewSPD creates a new SPD from wavelength and value vectors.
func NewSPD(wavelengths, values vecTypes.Vector) SPD {
	wl := wavelengths.View().(vec.Vector)
	vals := values.View().(vec.Vector)

	if wl.Len() != vals.Len() {
		panic("wavelengths and values must have the same length")
	}

	// Create 2-row matrix: row0 = wavelengths, row1 = values
	m := mat.New(2, wl.Len())
	m.SetRow(0, wl)
	m.SetRow(1, vals)

	return SPD{Matrix: m}
}

// Wavelengths returns the wavelength vector (row 0) from an SPD.
func (spd SPD) Wavelengths() vec.Vector {
	if spd.Matrix == nil {
		return nil
	}
	return spd.Matrix.Row(0).(vec.Vector)
}

// Values returns the values vector (row 1) from an SPD.
func (spd SPD) Values() vec.Vector {
	if spd.Matrix == nil {
		return nil
	}
	return spd.Matrix.Row(1).(vec.Vector)
}

// Len returns the number of wavelength/value pairs in an SPD.
func (spd SPD) Len() int {
	if spd.Matrix == nil || spd.Matrix.Rows() < 2 {
		return 0
	}
	return spd.Matrix.Cols()
}

// Interpolate resamples the SPD to new wavelengths using linear interpolation.
// The SPD is interpolated to match the target wavelengths (CMF/illuminant have higher resolution).
func (spd SPD) Interpolate(targetWavelengths vecTypes.Vector) SPD {
	targetWl := targetWavelengths.View().(vec.Vector)
	if targetWl.Len() == 0 {
		return SPD{Matrix: mat.New(2, 0)}
	}

	if spd.Matrix == nil {
		return SPD{Matrix: mat.New(2, targetWl.Len())}
	}

	sourceWl := spd.Wavelengths()
	sourceVals := spd.Values()

	if sourceWl.Len() == 0 {
		return SPD{Matrix: mat.New(2, targetWl.Len())}
	}
	if sourceWl.Len() == 1 {
		// Constant value
		newVals := vec.New(targetWl.Len())
		for i := 0; i < targetWl.Len(); i++ {
			newVals[i] = sourceVals[0]
		}
		return NewSPD(targetWl, newVals)
	}

	// Linear interpolation
	newValues := vec.New(targetWl.Len())

	for i := 0; i < targetWl.Len(); i++ {
		targetWlVal := targetWl[i]

		// Find surrounding points
		idx := 0
		for idx < sourceWl.Len()-1 && sourceWl[idx+1] < targetWlVal {
			idx++
		}

		if idx == sourceWl.Len()-1 {
			// Extrapolate: use last value
			newValues[i] = sourceVals[idx]
		} else if targetWlVal <= sourceWl[0] {
			// Extrapolate: use first value
			newValues[i] = sourceVals[0]
		} else {
			// Interpolate between idx and idx+1
			wl0 := sourceWl[idx]
			wl1 := sourceWl[idx+1]
			val0 := sourceVals[idx]
			val1 := sourceVals[idx+1]

			t := (targetWlVal - wl0) / (wl1 - wl0)
			newValues[i] = interpolation.Lerp(val0, val1, t)
		}
	}

	return NewSPD(targetWl, newValues)
}

// Wavelengths returns the wavelengths as an SPD (row 0 + row 0).
func (cmf ObserverCMF) Wavelengths() SPD {
	if cmf.Matrix == nil {
		return SPD{}
	}
	wl := cmf.Matrix.Row(0).(vec.Vector)
	return NewSPD(wl, wl)
}

// WavelengthsValues returns the wavelength vector (row 0) from a CMF.
func (cmf ObserverCMF) WavelengthsValues() vec.Vector {
	if cmf.Matrix == nil {
		return nil
	}
	return cmf.Matrix.Row(0).(vec.Vector)
}

// XBar returns the XBar as an SPD (wavelengths row + XBar row 1).
func (cmf ObserverCMF) XBar() SPD {
	if cmf.Matrix == nil {
		return SPD{}
	}
	wl := cmf.Matrix.Row(0).(vec.Vector)
	xBar := cmf.Matrix.Row(1).(vec.Vector)
	return NewSPD(wl, xBar)
}

// XBarValues returns the XBar vector (row 1) from a CMF.
func (cmf ObserverCMF) XBarValues() vec.Vector {
	if cmf.Matrix == nil {
		return nil
	}
	return cmf.Matrix.Row(1).(vec.Vector)
}

// YBar returns the YBar as an SPD (wavelengths row + YBar row 2).
func (cmf ObserverCMF) YBar() SPD {
	if cmf.Matrix == nil {
		return SPD{}
	}
	wl := cmf.Matrix.Row(0).(vec.Vector)
	yBar := cmf.Matrix.Row(2).(vec.Vector)
	return NewSPD(wl, yBar)
}

// YBarValues returns the YBar vector (row 2) from a CMF.
func (cmf ObserverCMF) YBarValues() vec.Vector {
	if cmf.Matrix == nil {
		return nil
	}
	return cmf.Matrix.Row(2).(vec.Vector)
}

// ZBar returns the ZBar as an SPD (wavelengths row + ZBar row 3).
func (cmf ObserverCMF) ZBar() SPD {
	if cmf.Matrix == nil {
		return SPD{}
	}
	wl := cmf.Matrix.Row(0).(vec.Vector)
	zBar := cmf.Matrix.Row(3).(vec.Vector)
	return NewSPD(wl, zBar)
}

// ZBarValues returns the ZBar vector (row 3) from a CMF.
func (cmf ObserverCMF) ZBarValues() vec.Vector {
	if cmf.Matrix == nil {
		return nil
	}
	return cmf.Matrix.Row(3).(vec.Vector)
}

// Len returns the number of wavelength/CMF pairs in a CMF.
func (cmf ObserverCMF) Len() int {
	if cmf.Matrix == nil || cmf.Matrix.Rows() < 4 {
		return 0
	}
	return cmf.Matrix.Cols()
}

// CalibrationPoint represents a calibration point: index -> wavelength.
type CalibrationPoint struct {
	Index      int     // Index in the SPD (0-based)
	Wavelength float32 // Actual wavelength in nanometers
}

// Calibrate calibrates the SPD using known wavelength calibration points.
// Takes variadic float32 values that must be even in length, interpreted as (index, wavelength) pairs.
// The SPD must already be initialized with wavelengths and values rows.
//
// Calibration process:
//  1. Uses the calibration points to calculate wavelengths for all indices using linear interpolation
//  2. Updates the wavelength row of the SPD with the calibrated wavelengths
//  3. The values row remains unchanged (values stay at their indices, wavelengths are corrected)
//
// Example: Calibrate(0, 400.0, 100, 700.0) calibrates so that index 0 = 400nm, index 100 = 700nm,
// and wavelengths for intermediate indices are linearly interpolated.
func (spd SPD) Calibrate(pairs ...float32) (SPD, error) {
	if spd.Matrix == nil {
		return SPD{}, fmt.Errorf("SPD matrix cannot be nil")
	}

	if len(pairs) == 0 {
		return SPD{}, fmt.Errorf("pairs cannot be empty")
	}

	if len(pairs)%2 != 0 {
		return SPD{}, fmt.Errorf("pairs length must be even (index, wavelength pairs), got %d", len(pairs))
	}

	if spd.Len() == 0 {
		return SPD{}, fmt.Errorf("SPD cannot be empty")
	}

	// Parse calibration points from pairs
	calPoints := make([]CalibrationPoint, 0, len(pairs)/2)
	for i := 0; i < len(pairs); i += 2 {
		idx := int(pairs[i])
		wl := pairs[i+1]

		// Validate index
		if idx < 0 || idx >= spd.Len() {
			return SPD{}, fmt.Errorf("calibration point index %d out of range [0, %d)", idx, spd.Len())
		}

		calPoints = append(calPoints, CalibrationPoint{
			Index:      idx,
			Wavelength: wl,
		})
	}

	if len(calPoints) < 2 {
		return SPD{}, fmt.Errorf("at least 2 calibration points required, got %d", len(calPoints))
	}

	// Sort calibration points by index
	sort.Slice(calPoints, func(i, j int) bool {
		return calPoints[i].Index < calPoints[j].Index
	})

	// Check for duplicate indices
	for i := 0; i < len(calPoints)-1; i++ {
		if calPoints[i].Index == calPoints[i+1].Index {
			return SPD{}, fmt.Errorf("duplicate calibration point at index %d", calPoints[i].Index)
		}
	}

	// Calculate wavelengths for all indices using cubic Catmull-Rom spline interpolation
	calibratedWavelengths := vec.New(spd.Len())
	values := spd.Values()

	// Handle indices before first calibration point (extrapolate using linear)
	firstIdx := calPoints[0].Index
	firstWl := calPoints[0].Wavelength
	if firstIdx > 0 {
		// Use slope from first two points for linear extrapolation
		if len(calPoints) >= 2 {
			secondIdx := calPoints[1].Index
			secondWl := calPoints[1].Wavelength
			slope := (secondWl - firstWl) / float32(secondIdx-firstIdx)

			for i := 0; i < firstIdx; i++ {
				calibratedWavelengths[i] = firstWl + slope*float32(i-firstIdx)
			}
		} else {
			// Single point - constant wavelength (shouldn't happen due to check above)
			for i := 0; i < firstIdx; i++ {
				calibratedWavelengths[i] = firstWl
			}
		}
	}

	// Set calibration point wavelengths directly
	for _, cp := range calPoints {
		calibratedWavelengths[cp.Index] = cp.Wavelength
	}

	// Interpolate between calibration points using cubic Catmull-Rom spline
	for pIdx := 0; pIdx < len(calPoints)-1; pIdx++ {
		idx0 := calPoints[pIdx].Index
		wl0 := calPoints[pIdx].Wavelength
		idx1 := calPoints[pIdx+1].Index
		wl1 := calPoints[pIdx+1].Wavelength

		// Get control points for Catmull-Rom spline (p1, p2, p3, p4)
		// For Catmull-Rom: S(t) interpolates between p2 and p3 using p1 and p4 as control
		var p1, p2, p3, p4 float32

		// p2 = wavelength at idx0, p3 = wavelength at idx1
		p2 = wl0
		p3 = wl1

		// Get p1 (point before idx0)
		if pIdx > 0 {
			p1 = calPoints[pIdx-1].Wavelength
		} else {
			// First segment: use first point as p1
			p1 = wl0
		}

		// Get p4 (point after idx1)
		if pIdx+2 < len(calPoints) {
			p4 = calPoints[pIdx+2].Wavelength
		} else {
			// Last segment: use last point as p4
			p4 = wl1
		}

		// Interpolate between idx0 and idx1 using cubic Catmull-Rom spline
		for i := idx0 + 1; i < idx1; i++ {
			// Normalized parameter t in [0, 1] for the segment [idx0, idx1]
			t := float32(i-idx0) / float32(idx1-idx0)
			calibratedWavelengths[i] = interpolation.CubicCatmulRomSpline1D(p1, p2, p3, p4, t)
		}
	}

	// Handle indices after last calibration point (extrapolate using linear)
	lastIdx := calPoints[len(calPoints)-1].Index
	lastWl := calPoints[len(calPoints)-1].Wavelength
	if lastIdx < spd.Len()-1 {
		// Use slope from last two points for linear extrapolation
		if len(calPoints) >= 2 {
			secondLastIdx := calPoints[len(calPoints)-2].Index
			secondLastWl := calPoints[len(calPoints)-2].Wavelength
			slope := (lastWl - secondLastWl) / float32(lastIdx-secondLastIdx)

			for i := lastIdx + 1; i < spd.Len(); i++ {
				calibratedWavelengths[i] = lastWl + slope*float32(i-lastIdx)
			}
		} else {
			// Single point - constant wavelength (shouldn't happen)
			for i := lastIdx + 1; i < spd.Len(); i++ {
				calibratedWavelengths[i] = lastWl
			}
		}
	}

	// Create new SPD with calibrated wavelengths and original values
	return NewSPD(calibratedWavelengths, values), nil
}

// Reconstruct reconstructs the SPD values from measurements of multiple photodetectors with known spectral responses.
// The SPD must already be initialized with wavelengths row and zeroed values row.
// This solves the inverse problem: given sensor responses R and sensor spectral sensitivities S,
// find the stimulus SPD X such that R = S · X.
//
// The problem is formulated as a linear system: R = S · X
// where:
//   - R is a vector of sensor readings (length = num_sensors)
//   - S is a matrix of sensor responses (num_sensors × num_wavelengths)
//   - X is the unknown stimulus SPD values (length = num_wavelengths)
//
// This is solved using pseudo-inverse: X = S^+ · R
//
// Parameters:
//   - sensorResponses: Array of sensor responses with their measured readings
//   - useDampedLS: If true, use damped least squares (Tikhonov regularization)
//   - lambda: Regularization parameter for damped least squares (only used if useDampedLS=true)
//
// Returns error if reconstruction fails.
func (spd SPD) Reconstruct(sensorResponses []SensorResponse, useDampedLS bool, lambda float32) error {
	if spd.Matrix == nil {
		return fmt.Errorf("SPD matrix cannot be nil")
	}

	if len(sensorResponses) == 0 {
		return fmt.Errorf("at least one sensor response required")
	}

	targetWlVec := spd.Wavelengths()
	if targetWlVec.Len() == 0 {
		return fmt.Errorf("SPD wavelengths must not be empty")
	}

	targetVals := spd.Values()
	if targetVals.Len() != targetWlVec.Len() {
		return fmt.Errorf("SPD values length (%d) must match wavelengths length (%d)", targetVals.Len(), targetWlVec.Len())
	}

	targetWl := vecTypes.Vector(targetWlVec)

	// Build sensor response matrix S (num_sensors × num_wavelengths)
	// Each row is a sensor's spectral response interpolated to target wavelengths
	numSensors := len(sensorResponses)
	numWavelengths := targetWlVec.Len()

	S := mat.New(numSensors, numWavelengths)
	readings := vec.New(numSensors)

	for i, sensor := range sensorResponses {
		if sensor.Response == nil {
			return fmt.Errorf("sensor %d (%s) has nil response", i, sensor.Name)
		}

		// Interpolate sensor response to target wavelengths
		sensorSPD := SPD{Matrix: sensor.Response}
		interpResponse := sensorSPD.Interpolate(targetWl)

		// Set row i of matrix S to the interpolated response
		S.SetRow(i, interpResponse.Values())

		// Store reading
		readings[i] = sensor.Reading
	}

	// Solve for X: X = S^+ · R
	// where S^+ is the pseudo-inverse of S

	var X vec.Vector

	if useDampedLS {
		// Use damped least squares (Tikhonov regularization)
		// X = (S^T · S + λ²I)^(-1) · S^T · R

		// Compute S^T
		ST := mat.New(numWavelengths, numSensors)
		ST.Transpose(S)

		// Compute S^T · S
		STS := mat.New(numWavelengths, numWavelengths)
		STS.Mul(ST, S)

		// Add regularization: STS + λ²I
		lambdaSq := lambda * lambda
		for i := 0; i < numWavelengths; i++ {
			STS[i][i] += lambdaSq
		}

		// Compute (STS + λ²I)^(-1)
		STSInv := mat.New(numWavelengths, numWavelengths)
		if err := STS.Inverse(STSInv); err != nil {
			return fmt.Errorf("failed to invert regularized matrix: %w", err)
		}

		// Compute X = (STS + λ²I)^(-1) · S^T · R
		ST_R := vec.New(numWavelengths)
		ST.MulVec(readings, ST_R)

		X = vec.New(numWavelengths)
		STSInv.MulVec(ST_R, X)
	} else {
		// Use standard pseudo-inverse
		// Compute S^+
		SPseudo := mat.New(numWavelengths, numSensors)
		if err := S.PseudoInverse(SPseudo); err != nil {
			return fmt.Errorf("failed to compute pseudo-inverse: %w", err)
		}

		// Compute X = S^+ · R
		X = vec.New(numWavelengths)
		SPseudo.MulVec(readings, X)
	}

	// Ensure non-negative values (physical constraint: SPD cannot be negative)
	for i := 0; i < X.Len(); i++ {
		if X[i] < 0 {
			X[i] = 0
		}
	}

	// Fill in the values row of the SPD
	spd.Matrix.SetRow(1, X)

	return nil
}

// ReconstructWeighted reconstructs the SPD using weighted least squares, accounting for measurement uncertainty.
// Measurements with higher uncertainty (lower confidence) are given less weight in the reconstruction.
//
// Weighted least squares: minimize ||W · (S · X - R)||²
// where W is a diagonal weight matrix with weights w_i = 1 / uncertainty_i²
//
// The solution is: X = (S^T · W² · S)^(-1) · S^T · W² · R
//
// Parameters:
//   - sensorResponses: Array of sensor responses with readings
//   - weights: Weight vector (one per sensor). Higher weight = more trust in measurement. Weights are normalized if needed.
//   - useDampedLS: If true, use weighted damped least squares (Tikhonov regularization)
//   - lambda: Regularization parameter for damped least squares (only used if useDampedLS=true)
//
// Returns error if reconstruction fails.
//
// Note: If weights is nil or all zeros, falls back to unweighted reconstruction.
func (spd SPD) ReconstructWeighted(sensorResponses []SensorResponse, weights vecTypes.Vector, useDampedLS bool, lambda float32) error {
	if spd.Matrix == nil {
		return fmt.Errorf("SPD matrix cannot be nil")
	}

	if len(sensorResponses) == 0 {
		return fmt.Errorf("at least one sensor response required")
	}

	targetWlVec := spd.Wavelengths()
	if targetWlVec.Len() == 0 {
		return fmt.Errorf("SPD wavelengths must not be empty")
	}

	targetVals := spd.Values()
	if targetVals.Len() != targetWlVec.Len() {
		return fmt.Errorf("SPD values length (%d) must match wavelengths length (%d)", targetVals.Len(), targetWlVec.Len())
	}

	targetWl := vecTypes.Vector(targetWlVec)

	// Build sensor response matrix S (num_sensors × num_wavelengths)
	numSensors := len(sensorResponses)
	numWavelengths := targetWlVec.Len()

	S := mat.New(numSensors, numWavelengths)
	readings := vec.New(numSensors)

	for i, sensor := range sensorResponses {
		if sensor.Response == nil {
			return fmt.Errorf("sensor %d (%s) has nil response", i, sensor.Name)
		}

		// Interpolate sensor response to target wavelengths
		sensorSPD := SPD{Matrix: sensor.Response}
		interpResponse := sensorSPD.Interpolate(targetWl)

		// Set row i of matrix S to the interpolated response
		S.SetRow(i, interpResponse.Values())

		// Store reading
		readings[i] = sensor.Reading
	}

	// Process weights
	w := weights
	if w != nil {
		wVec := w.View().(vec.Vector)
		if wVec != nil && wVec.Len() == numSensors {
			// Check if all weights are zero (unweighted)
			hasNonZeroWeight := false
			for i := 0; i < numSensors; i++ {
				if wVec[i] > 0 {
					hasNonZeroWeight = true
					break
				}
			}
			if !hasNonZeroWeight {
				w = nil // Fall back to unweighted
			}
		} else {
			w = nil // Invalid weights, fall back to unweighted
		}
	}

	var X vec.Vector

	if w == nil {
		// Fall back to unweighted reconstruction
		return spd.Reconstruct(sensorResponses, useDampedLS, lambda)
	}

	// Weighted least squares
	wVec := w.View().(vec.Vector)

	// Normalize weights (optional - improves numerical stability)
	// Find max weight for normalization
	maxWeight := float32(0)
	for i := 0; i < numSensors; i++ {
		if wVec[i] > maxWeight {
			maxWeight = wVec[i]
		}
	}
	if maxWeight > 0 {
		// Normalize weights (keep relative ratios)
		for i := 0; i < numSensors; i++ {
			wVec[i] /= maxWeight
		}
	}

	// Build weight matrix W² (diagonal matrix: W²_ii = w_i²)
	W2 := mat.New(numSensors, numSensors)
	for i := 0; i < numSensors; i++ {
		wSq := wVec[i] * wVec[i]
		W2[i][i] = wSq
	}

	// Compute S^T
	ST := mat.New(numWavelengths, numSensors)
	ST.Transpose(S)

	// Compute S^T · W²
	ST_W2 := mat.New(numWavelengths, numSensors)
	ST_W2.Mul(ST, W2)

	// Compute S^T · W² · S
	ST_W2_S := mat.New(numWavelengths, numWavelengths)
	ST_W2_S.Mul(ST_W2, S)

	// Compute S^T · W² · R
	W2_R := vec.New(numSensors)
	for i := 0; i < numSensors; i++ {
		W2_R[i] = W2[i][i] * readings[i]
	}
	ST_W2_R := vec.New(numWavelengths)
	ST.MulVec(W2_R, ST_W2_R)

	if useDampedLS {
		// Weighted damped least squares: (S^T · W² · S + λ²I)^(-1) · S^T · W² · R
		lambdaSq := lambda * lambda
		for i := 0; i < numWavelengths; i++ {
			ST_W2_S[i][i] += lambdaSq
		}

		// Compute (S^T · W² · S + λ²I)^(-1)
		ST_W2_S_Inv := mat.New(numWavelengths, numWavelengths)
		if err := ST_W2_S.Inverse(ST_W2_S_Inv); err != nil {
			return fmt.Errorf("failed to invert weighted regularized matrix: %w", err)
		}

		// Compute X = (S^T · W² · S + λ²I)^(-1) · S^T · W² · R
		X = vec.New(numWavelengths)
		ST_W2_S_Inv.MulVec(ST_W2_R, X)
	} else {
		// Weighted least squares: X = (S^T · W² · S)^(-1) · S^T · W² · R
		ST_W2_S_Inv := mat.New(numWavelengths, numWavelengths)
		if err := ST_W2_S.Inverse(ST_W2_S_Inv); err != nil {
			return fmt.Errorf("failed to invert weighted matrix: %w", err)
		}

		// Compute X = (S^T · W² · S)^(-1) · S^T · W² · R
		X = vec.New(numWavelengths)
		ST_W2_S_Inv.MulVec(ST_W2_R, X)
	}

	// Ensure non-negative values (physical constraint: SPD cannot be negative)
	for i := 0; i < X.Len(); i++ {
		if X[i] < 0 {
			X[i] = 0
		}
	}

	// Fill in the values row of the SPD
	spd.Matrix.SetRow(1, X)

	return nil
}

// parseCSV parses a CSV file with wavelength,value or wavelength,x,y,z format.
// Returns wavelengths and a slice of value vectors (one per column after wavelength).
func parseCSV(data string) (vec.Vector, []vec.Vector, error) {
	reader := csv.NewReader(strings.NewReader(data))
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse CSV: %w", err)
	}

	if len(records) == 0 {
		return nil, nil, fmt.Errorf("CSV file is empty")
	}

	// Skip header if present (first line contains non-numeric data)
	startIdx := 0
	if len(records) > 0 {
		_, err := strconv.ParseFloat(records[0][0], 32)
		if err != nil {
			startIdx = 1 // Skip header
		}
	}

	numCols := len(records[startIdx])
	if numCols < 2 {
		return nil, nil, fmt.Errorf("CSV must have at least 2 columns (wavelength, value)")
	}

	wavelengths := vec.New(len(records) - startIdx)
	numValueCols := numCols - 1
	values := make([]vec.Vector, numValueCols)
	for i := range values {
		values[i] = vec.New(len(records) - startIdx)
	}

	for i, record := range records[startIdx:] {
		if len(record) != numCols {
			return nil, nil, fmt.Errorf("inconsistent column count at row %d", i+startIdx+1)
		}

		wl, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid wavelength at row %d: %w", i+startIdx+1, err)
		}
		wavelengths[i] = float32(wl)

		for j := 0; j < numValueCols; j++ {
			val, err := strconv.ParseFloat(record[j+1], 32)
			if err != nil {
				return nil, nil, fmt.Errorf("invalid value at row %d, column %d: %w", i+startIdx+1, j+2, err)
			}
			values[j][i] = float32(val)
		}
	}

	return wavelengths, values, nil
}
