package colorscience

import (
	"github.com/chewxy/math32"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Peak represents a detected peak in an SPD.
type Peak struct {
	Index      int     // Index in the SPD
	Wavelength float32 // Wavelength at the peak
	Value      float32 // Peak value (intensity)
	Prominence float32 // Prominence of the peak (height above surrounding baseline)
}

// Valley represents a detected valley in an SPD.
type Valley struct {
	Index      int     // Index in the SPD
	Wavelength float32 // Wavelength at the valley
	Value      float32 // Valley value (intensity)
	Prominence float32 // Prominence of the valley (depth below surrounding baseline)
}

// Peaks detects local maxima (peaks) in the SPD.
// Peaks are detected where values are higher than their neighbors.
// Only peaks with prominence >= minProminence are returned.
// Threshold filters out peaks below a minimum value.
//
// Parameters:
//   - threshold: Minimum peak value to consider (0 = no threshold)
//   - minProminence: Minimum prominence for a peak to be included (0 = no prominence filter)
//
// Returns: slice of detected peaks, sorted by wavelength
func (spd SPD) Peaks(threshold, minProminence float32) []Peak {
	if spd.Matrix == nil {
		return nil
	}

	vals := spd.Values()
	wl := spd.Wavelengths()

	if vals == nil || wl == nil || vals.Len() < 3 {
		return nil
	}

	var peaks []Peak

	// Detect local maxima
	for i := 1; i < vals.Len()-1; i++ {
		// Check if this is a local maximum
		if vals[i] > vals[i-1] && vals[i] > vals[i+1] {
			// Check threshold
			if threshold > 0 && vals[i] < threshold {
				continue
			}

			// Calculate prominence (simplified: height above average of neighbors)
			avgNeighbors := (vals[i-1] + vals[i+1]) / 2.0
			prominence := vals[i] - avgNeighbors

			if minProminence > 0 && prominence < minProminence {
				continue
			}

			peaks = append(peaks, Peak{
				Index:      i,
				Wavelength: wl[i],
				Value:      vals[i],
				Prominence: prominence,
			})
		}
	}

	return peaks
}

// Valleys detects local minima (valleys) in the SPD.
// Valleys are detected where values are lower than their neighbors.
// Only valleys with prominence >= minProminence are returned.
// Threshold filters out valleys above a maximum value.
//
// Parameters:
//   - threshold: Maximum valley value to consider (0 = no threshold, use very large value for no filter)
//   - minProminence: Minimum prominence for a valley to be included (0 = no prominence filter)
//
// Returns: slice of detected valleys, sorted by wavelength
func (spd SPD) Valleys(threshold, minProminence float32) []Valley {
	if spd.Matrix == nil {
		return nil
	}

	vals := spd.Values()
	wl := spd.Wavelengths()

	if vals == nil || wl == nil || vals.Len() < 3 {
		return nil
	}

	var valleys []Valley

	// Detect local minima
	for i := 1; i < vals.Len()-1; i++ {
		// Check if this is a local minimum
		if vals[i] < vals[i-1] && vals[i] < vals[i+1] {
			// Check threshold (if threshold is 0, use a very large value to disable)
			maxThreshold := float32(1e10)
			if threshold > 0 {
				maxThreshold = threshold
			}
			if vals[i] > maxThreshold {
				continue
			}

			// Calculate prominence (simplified: depth below average of neighbors)
			avgNeighbors := (vals[i-1] + vals[i+1]) / 2.0
			prominence := avgNeighbors - vals[i]

			if minProminence > 0 && prominence < minProminence {
				continue
			}

			valleys = append(valleys, Valley{
				Index:      i,
				Wavelength: wl[i],
				Value:      vals[i],
				Prominence: prominence,
			})
		}
	}

	return valleys
}

// DetectCalibrationPoints matches this SPD to a reference SPD and returns calibration points.
// Uses correlation-based matching to find corresponding features between the two spectra.
// Returns (index, wavelength) pairs that can be used with SPD.Calibrate().
//
// The algorithm:
//  1. Interpolates both SPDs to a common wavelength grid
//  2. Uses sliding window correlation to find best matches
//  3. Detects peaks in both spectra and matches them
//  4. Returns calibration points with confidence scores >= minConfidence
//
// Parameters:
//   - referenceSPD: The reference SPD with known wavelengths (e.g., D65 illuminant, any known spectrum)
//   - minConfidence: Minimum confidence score (0-1) for a match to be included
//
// Returns: slice of calibration points sorted by index in this SPD
func (spd SPD) DetectCalibrationPoints(referenceSPD SPD, minConfidence float32) []CalibrationPoint {
	if spd.Matrix == nil || referenceSPD.Matrix == nil {
		return nil
	}

	if spd.Len() < 2 || referenceSPD.Len() < 2 {
		return nil
	}

	// Interpolate both to common wavelength grid (use reference wavelengths as target)
	refWl := referenceSPD.Wavelengths()
	if refWl == nil || refWl.Len() == 0 {
		return nil
	}

	refWlVec := vecTypes.Vector(refWl)
	interpMeasured := spd.Interpolate(refWlVec)
	interpRef := referenceSPD.Interpolate(refWlVec)

	measVals := interpMeasured.Values()
	refVals := interpRef.Values()
	measWl := spd.Wavelengths()

	if measVals == nil || refVals == nil || measWl == nil {
		return nil
	}

	// Detect peaks in both spectra for feature matching
	measPeaks := spd.Peaks(0, 0)
	refPeaks := referenceSPD.Peaks(0, 0)

	if len(measPeaks) == 0 || len(refPeaks) == 0 {
		// Fallback: use correlation-based matching with sliding window
		return detectCalibrationByCorrelation(interpMeasured, referenceSPD, minConfidence)
	}

	// Match peaks between measured and reference spectra
	var calPoints []CalibrationPoint

	// Find corresponding peaks using cross-correlation around peak locations
	for _, measPeak := range measPeaks {
		bestRefPeak := -1
		bestCorrelation := float32(-1.0)

		// Search for matching reference peak
		for i, refPeak := range refPeaks {
			// Calculate correlation in a window around the peaks
			window := 10 // wavelength points around peak
			corr := correlationAroundPeak(interpMeasured, referenceSPD, measPeak.Index, refPeak.Index, window)
			if corr > bestCorrelation {
				bestCorrelation = corr
				bestRefPeak = i
			}
		}

		if bestCorrelation >= minConfidence && bestRefPeak >= 0 {
			refPeak := refPeaks[bestRefPeak]
			// Find original index in measured SPD
			measIdx := measPeak.Index
			if measIdx < measWl.Len() {
				calPoints = append(calPoints, CalibrationPoint{
					Index:      measIdx,
					Wavelength: refPeak.Wavelength,
				})
			}
		}
	}

	// If we didn't find enough points, use correlation-based matching as fallback
	if len(calPoints) < 2 {
		calPoints = detectCalibrationByCorrelation(interpMeasured, referenceSPD, minConfidence)
	}

	return calPoints
}

// DetectCalibrationPoints is a convenience wrapper for the SPD method.
// Matches a measured SPD to a reference SPD and returns calibration points.
//
// Deprecated: Use spd.DetectCalibrationPoints(referenceSPD, minConfidence) instead.
func DetectCalibrationPoints(measuredSPD, referenceSPD SPD, minConfidence float32) []CalibrationPoint {
	return measuredSPD.DetectCalibrationPoints(referenceSPD, minConfidence)
}

// correlationAroundPeak calculates correlation between two SPDs around given indices.
func correlationAroundPeak(spd1, spd2 SPD, idx1, idx2, window int) float32 {
	vals1 := spd1.Values()
	vals2 := spd2.Values()

	if vals1 == nil || vals2 == nil {
		return 0
	}

	// Ensure window doesn't go out of bounds
	start1 := idx1 - window
	end1 := idx1 + window
	start2 := idx2 - window
	end2 := idx2 + window

	if start1 < 0 {
		start1 = 0
	}
	if end1 > vals1.Len() {
		end1 = vals1.Len()
	}
	if start2 < 0 {
		start2 = 0
	}
	if end2 > vals2.Len() {
		end2 = vals2.Len()
	}

	len1 := end1 - start1
	len2 := end2 - start2
	if len1 != len2 || len1 == 0 {
		return 0
	}

	// Calculate Pearson correlation coefficient
	var sum1, sum2, sum1Sq, sum2Sq, sumProd float32
	for i := 0; i < len1; i++ {
		v1 := vals1[start1+i]
		v2 := vals2[start2+i]
		sum1 += v1
		sum2 += v2
		sum1Sq += v1 * v1
		sum2Sq += v2 * v2
		sumProd += v1 * v2
	}

	n := float32(len1)
	mean1 := sum1 / n
	mean2 := sum2 / n

	cov := (sumProd / n) - (mean1 * mean2)
	std1 := (sum1Sq/n - mean1*mean1)
	std2 := (sum2Sq/n - mean2*mean2)

	if std1 <= 0 || std2 <= 0 {
		return 0
	}

	corr := cov / (math32.Sqrt(std1) * math32.Sqrt(std2))
	if corr < 0 {
		corr = 0 // Return 0 for negative correlation
	}

	return corr
}

// detectCalibrationByCorrelation uses sliding window correlation to find calibration points.
func detectCalibrationByCorrelation(measuredSPD, referenceSPD SPD, minConfidence float32) []CalibrationPoint {
	measVals := measuredSPD.Values()
	refVals := referenceSPD.Values()
	measWl := measuredSPD.Wavelengths()
	refWl := referenceSPD.Wavelengths()

	if measVals == nil || refVals == nil || measWl == nil || refWl == nil {
		return nil
	}

	if measVals.Len() != refVals.Len() {
		return nil
	}

	var calPoints []CalibrationPoint

	// Sample points at regular intervals and use correlation to verify matches
	sampleInterval := measVals.Len() / 10 // Sample ~10 points
	if sampleInterval < 1 {
		sampleInterval = 1
	}

	window := 5 // Correlation window size
	for i := window; i < measVals.Len()-window; i += sampleInterval {
		// Calculate correlation around this point
		var sum1, sum2, sum1Sq, sum2Sq, sumProd float32
		for j := -window; j <= window; j++ {
			idx := i + j
			if idx < 0 || idx >= measVals.Len() {
				continue
			}
			v1 := measVals[idx]
			v2 := refVals[idx]
			sum1 += v1
			sum2 += v2
			sum1Sq += v1 * v1
			sum2Sq += v2 * v2
			sumProd += v1 * v2
		}

		n := float32(2*window + 1)
		mean1 := sum1 / n
		mean2 := sum2 / n

		cov := (sumProd / n) - (mean1 * mean2)
		std1 := (sum1Sq/n - mean1*mean1)
		std2 := (sum2Sq/n - mean2*mean2)

		if std1 > 0 && std2 > 0 {
			corr := cov / (math32.Sqrt(std1) * math32.Sqrt(std2))
			if corr >= minConfidence {
				calPoints = append(calPoints, CalibrationPoint{
					Index:      i,
					Wavelength: refWl[i],
				})
			}
		}
	}

	return calPoints
}
