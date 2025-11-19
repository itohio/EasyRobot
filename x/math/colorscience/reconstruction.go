package colorscience

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// SensorResponse represents a photodetector with its known spectral response.
type SensorResponse struct {
	Name     string          // Sensor name/identifier
	Response matTypes.Matrix // Spectral response of the sensor (wavelengths -> response) - must be 2-row SPD matrix
	Reading  float32         // Measured response value to the stimulus
}

// ReconstructSPDWithConstraints reconstructs SPD with additional constraints.
// This is a more advanced version that allows specifying constraints like:
// - Non-negativity (already enforced in SPD.Reconstruct)
// - Smoothness (can be added via regularization)
// - Known values at specific wavelengths
//
// For now, this is a convenience wrapper that creates an SPD with the target wavelengths and zeroed values,
// then calls SPD.Reconstruct().
// Future enhancements could add:
// - Smoothness regularization (second derivative penalty)
// - Known value constraints
// - Bounds on values
func ReconstructSPDWithConstraints(
	sensorResponses []SensorResponse,
	targetWavelengths vecTypes.Vector,
	useDampedLS bool,
	lambda float32,
	smoothnessWeight float32, // Weight for smoothness regularization (0 = no smoothing)
) (SPD, error) {
	targetWl := targetWavelengths.View().(vec.Vector)
	if targetWl.Len() == 0 {
		return SPD{}, fmt.Errorf("target wavelengths must not be empty")
	}

	// Create SPD with wavelengths and zeroed values
	zeroVals := vec.New(targetWl.Len())
	spd := NewSPD(targetWl, zeroVals)

	// Reconstruct values
	if err := spd.Reconstruct(sensorResponses, useDampedLS, lambda); err != nil {
		return SPD{}, err
	}

	return spd, nil
}

// SensorChannel represents a sensor channel with center wavelength and full width at half maximum (FWHM).
// This is useful for sensors like AS734x chips that only provide center wavelength and bandwidth.
type SensorChannel struct {
	Name        string  // Channel name/identifier
	CenterWL    float32 // Center wavelength in nanometers
	FWHM        float32 // Full width at half maximum (FWHM) in nanometers
	Reading     float32 // Measured response value (optional, can be set later)
	Uncertainty float32 // Measurement uncertainty/confidence (0 = no weight, >0 = 1/uncertainty² weight). If >0, used as weight in weighted least squares.
	FilterShape int     // Filter shape model: 0=Gaussian (default), 1=Super-Gaussian (n=4, steeper), 2=Super-Gaussian (n=6, very steep), 3=Boxcar (flat top, sharp sides)
	Asymmetry   float32 // Asymmetry factor (-1 to 1): 0=symmetric, >0=redshifted, <0=blueshifted. Default 0.
}

// FilterShapeModel defines the filter shape model type.
type FilterShapeModel int

const (
	FilterShapeGaussian       FilterShapeModel = 0 // Standard Gaussian (good general approximation)
	FilterShapeSuperGaussian4 FilterShapeModel = 1 // Super-Gaussian n=4 (steeper sides, better for interference filters)
	FilterShapeSuperGaussian6 FilterShapeModel = 2 // Super-Gaussian n=6 (very steep sides)
	FilterShapeBoxcar         FilterShapeModel = 3 // Boxcar (flat top, sharp sides, ideal filter)
)

// FilterSPDFromChannel creates an SPD representing a spectral filter response from center wavelength and FWHM.
// Supports multiple filter shapes optimized for interference filters like those in AS734x sensors.
//
// Filter shapes:
//   - Gaussian (default): Good general approximation, y = exp(-0.5 * ((λ - center) / sigma)^2)
//   - Super-Gaussian (n=4): Steeper sides, better matches interference filters
//   - Super-Gaussian (n=6): Very steep sides, closer to ideal filters
//   - Boxcar: Flat top with sharp sides (idealized)
//
// For interference filters (AS734x), Super-Gaussian (n=4) is typically most accurate.
//
// Parameters:
//   - channel: SensorChannel with center wavelength, FWHM, and optional shape/asymmetry
//   - wavelengths: Target wavelength vector to evaluate the filter at
//
// Returns: SPD representing the filter spectral response
func FilterSPDFromChannel(channel SensorChannel, wavelengths vecTypes.Vector) SPD {
	wl := wavelengths.View().(vec.Vector)
	if wl == nil || wl.Len() == 0 {
		return SPD{Matrix: mat.New(2, 0)}
	}

	// Convert FWHM to sigma based on filter shape
	// For Gaussian: sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355
	// For Super-Gaussian n: FWHM = 2 * sigma * (ln(2))^(1/n) * (2)^(1/n)
	var sigma float32
	shape := FilterShapeModel(channel.FilterShape)
	switch shape {
	case FilterShapeGaussian:
		const fwhmToSigma = 2.355
		sigma = channel.FWHM / fwhmToSigma
	case FilterShapeSuperGaussian4:
		// For n=4: FWHM = 2 * sigma * (ln(2))^(1/4) * (2)^(1/4)
		// (ln(2))^(1/4) ≈ 0.915, 2^(1/4) ≈ 1.189, product ≈ 1.088
		// So sigma ≈ FWHM / (2 * 1.088) ≈ FWHM / 2.176
		sigma = channel.FWHM / 2.176
	case FilterShapeSuperGaussian6:
		// For n=6: FWHM ≈ 2 * sigma * 1.105
		sigma = channel.FWHM / 2.21
	case FilterShapeBoxcar:
		// For boxcar, use half-width at half-maximum as "sigma"
		sigma = channel.FWHM / 2.0
	default:
		// Default to Gaussian
		sigma = channel.FWHM / 2.355
	}

	if sigma <= 0 {
		// Invalid FWHM, return zero response
		zeroVals := vec.New(wl.Len())
		return NewSPD(wl, zeroVals)
	}

	// Compute filter response at each wavelength
	values := vec.New(wl.Len())
	center := channel.CenterWL
	asymmetry := channel.Asymmetry
	if asymmetry < -1 {
		asymmetry = -1
	}
	if asymmetry > 1 {
		asymmetry = 1
	}

	for i := 0; i < wl.Len(); i++ {
		lambda := wl[i]
		diff := lambda - center

		// Apply asymmetry: shift center slightly based on direction
		effectiveCenter := center + asymmetry*sigma*0.1 // Small shift based on asymmetry

		var response float32
		switch shape {
		case FilterShapeGaussian:
			// Standard Gaussian
			twoSigmaSq := 2.0 * sigma * sigma
			exponent := -(diff * diff) / twoSigmaSq
			response = math32.Exp(exponent)

		case FilterShapeSuperGaussian4:
			// Super-Gaussian n=4: exp(-0.5 * |(λ - center) / sigma|^4)
			normalizedDiff := (lambda - effectiveCenter) / sigma
			exponent := -0.5 * normalizedDiff * normalizedDiff * normalizedDiff * normalizedDiff
			response = math32.Exp(exponent)

		case FilterShapeSuperGaussian6:
			// Super-Gaussian n=6: exp(-0.5 * |(λ - center) / sigma|^6)
			normalizedDiff := (lambda - effectiveCenter) / sigma
			exponent := -0.5 * normalizedDiff * normalizedDiff * normalizedDiff * normalizedDiff * normalizedDiff * normalizedDiff
			response = math32.Exp(exponent)

		case FilterShapeBoxcar:
			// Boxcar: flat top, sharp sides
			halfWidth := channel.FWHM / 2.0
			if math32.Abs(lambda-effectiveCenter) <= halfWidth {
				response = 1.0
			} else {
				// Sharp rolloff (use very steep Gaussian for smoothness)
				normalizedDiff := (lambda - effectiveCenter) / (halfWidth * 0.1) // Very narrow rolloff
				exponent := -normalizedDiff * normalizedDiff
				response = math32.Exp(exponent)
			}

		default:
			// Default to Gaussian
			twoSigmaSq := 2.0 * sigma * sigma
			exponent := -(diff * diff) / twoSigmaSq
			response = math32.Exp(exponent)
		}

		values[i] = response
	}

	return NewSPD(wl, values)
}

// GaussianSPDFromChannel is a convenience wrapper for FilterSPDFromChannel using Gaussian shape.
// Deprecated: Use FilterSPDFromChannel with FilterShapeGaussian instead for better control.
func GaussianSPDFromChannel(channel SensorChannel, wavelengths vecTypes.Vector) SPD {
	channel.FilterShape = int(FilterShapeGaussian)
	return FilterSPDFromChannel(channel, wavelengths)
}

// SensorChannelsToResponses converts SensorChannel specifications with readings to SensorResponse objects.
// Each channel's filter response SPD (Gaussian, Super-Gaussian, or Boxcar) is evaluated at the target wavelengths.
//
// Parameters:
//   - channels: Array of SensorChannel with center wavelength, FWHM, readings, and optional shape/uncertainty
//   - targetWavelengths: Wavelength vector to evaluate the filter responses at
//
// Returns: Array of SensorResponse ready for use with SPD.Reconstruct() or SPD.ReconstructWeighted()
func SensorChannelsToResponses(channels []SensorChannel, targetWavelengths vecTypes.Vector) []SensorResponse {
	if len(channels) == 0 {
		return nil
	}

	responses := make([]SensorResponse, len(channels))
	targetWl := targetWavelengths.View().(vec.Vector)

	for i, channel := range channels {
		// Create filter SPD for this channel (Gaussian, Super-Gaussian, or Boxcar)
		filterSPD := FilterSPDFromChannel(channel, targetWl)

		responses[i] = SensorResponse{
			Name:     channel.Name,
			Response: filterSPD.Matrix,
			Reading:  channel.Reading,
		}
	}

	return responses
}

// ReconstructFromChannels reconstructs an SPD from sensor channels specified by center wavelength and FWHM.
// This is a convenience wrapper that:
// 1. Creates filter SPD responses from channel specifications (Gaussian, Super-Gaussian, or Boxcar)
// 2. Converts to SensorResponse objects
// 3. Calls SPD.Reconstruct() or SPD.ReconstructWeighted() based on whether uncertainties are provided
//
// This is useful for sensors like AS734x chips that only provide center wavelength and bandwidth information.
//
// Filter shape recommendations for AS734x:
//   - FilterShapeSuperGaussian4 (recommended): Best matches interference filter characteristics
//   - FilterShapeSuperGaussian6: For very steep-sided filters
//   - FilterShapeGaussian: Simple approximation (faster, less accurate)
//   - FilterShapeBoxcar: Idealized filter (sharp sides, flat top)
//
// If channels have Uncertainty > 0, weighted least squares is used automatically.
//
// Parameters:
//   - spd: SPD to reconstruct (must have wavelengths initialized, values will be filled)
//   - channels: Array of SensorChannel with center wavelength, FWHM, readings, and optional shape/uncertainty
//   - useDampedLS: If true, use damped least squares (Tikhonov regularization)
//   - lambda: Regularization parameter for damped least squares (only used if useDampedLS=true)
//
// Returns error if reconstruction fails.
//
// Example for AS7341 (recommended - using Super-Gaussian for accuracy):
//
//	channels := []colorscience.SensorChannel{
//	    {Name: "F1", CenterWL: 415, FWHM: 20, Reading: 1000, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    {Name: "F2", CenterWL: 445, FWHM: 20, Reading: 1500, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    {Name: "F3", CenterWL: 480, FWHM: 20, Reading: 1200, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    // ... more channels
//	}
//	spd := colorscience.NewSPD(wavelengths, vec.New(len(wavelengths)))
//	err := spd.ReconstructFromChannels(channels, true, 0.01) // Use damped LS for stability
//
// Example with confidence intervals (weighted reconstruction):
//
//	channels := []colorscience.SensorChannel{
//	    {Name: "F1", CenterWL: 415, FWHM: 20, Reading: 1000, Uncertainty: 0.05, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    {Name: "F2", CenterWL: 445, FWHM: 20, Reading: 1500, Uncertainty: 0.03, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    // Uncertainty = confidence interval as fraction (0.05 = 5% uncertainty)
//	}
func (spd SPD) ReconstructFromChannels(channels []SensorChannel, useDampedLS bool, lambda float32) error {
	if spd.Matrix == nil {
		return fmt.Errorf("SPD matrix cannot be nil")
	}

	targetWl := spd.Wavelengths()
	if targetWl == nil || targetWl.Len() == 0 {
		return fmt.Errorf("SPD must have wavelengths initialized")
	}

	// Convert channels to SensorResponse objects
	sensorResponses := SensorChannelsToResponses(channels, vecTypes.Vector(targetWl))

	// Check if any channels have uncertainty (for weighted reconstruction)
	hasUncertainty := false
	weights := vec.New(len(channels))
	for i, channel := range channels {
		if channel.Uncertainty > 0 {
			hasUncertainty = true
			// Weight = 1 / uncertainty² (higher uncertainty = lower weight)
			weights[i] = 1.0 / (channel.Uncertainty * channel.Uncertainty)
		} else {
			weights[i] = 1.0 // Equal weight if no uncertainty specified
		}
	}

	if hasUncertainty {
		// Use weighted reconstruction
		return spd.ReconstructWeighted(sensorResponses, weights, useDampedLS, lambda)
	}

	// Use standard (unweighted) reconstruction
	return spd.Reconstruct(sensorResponses, useDampedLS, lambda)
}

// ReconstructSPDFromChannels reconstructs an SPD from sensor channels specified by center wavelength and FWHM.
// This is a convenience wrapper that creates a new SPD with the target wavelengths and zeroed values,
// then calls ReconstructFromChannels().
//
// Filter shape recommendations:
//   - FilterShapeSuperGaussian4 (recommended for AS734x): Best accuracy for interference filters
//   - FilterShapeSuperGaussian6: For very steep-sided filters
//   - FilterShapeGaussian: Simple approximation (default, less accurate for interference filters)
//
// Parameters:
//   - channels: Array of SensorChannel with center wavelength, FWHM, readings, and optional shape/uncertainty
//   - targetWavelengths: Wavelength vector for the reconstructed SPD
//   - useDampedLS: If true, use damped least squares (Tikhonov regularization) - recommended for stability
//   - lambda: Regularization parameter for damped least squares (typically 0.01-0.1, only used if useDampedLS=true)
//
// Returns: Reconstructed SPD, error
//
// Example for AS7341 (recommended - using Super-Gaussian n=4):
//
//	wavelengths := vec.NewFrom(350.0, 360.0, 370.0, ..., 1000.0)
//	channels := []colorscience.SensorChannel{
//	    {Name: "F1", CenterWL: 415, FWHM: 20, Reading: 1000, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    {Name: "F2", CenterWL: 445, FWHM: 20, Reading: 1500, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
//	    // ... more channels
//	}
//	spd, err := colorscience.ReconstructSPDFromChannels(channels, wavelengths, true, 0.01)
func ReconstructSPDFromChannels(
	channels []SensorChannel,
	targetWavelengths vecTypes.Vector,
	useDampedLS bool,
	lambda float32,
) (SPD, error) {
	targetWl := targetWavelengths.View().(vec.Vector)
	if targetWl.Len() == 0 {
		return SPD{}, fmt.Errorf("target wavelengths must not be empty")
	}

	// Create SPD with wavelengths and zeroed values
	zeroVals := vec.New(targetWl.Len())
	spd := NewSPD(targetWl, zeroVals)

	// Reconstruct from channels
	if err := spd.ReconstructFromChannels(channels, useDampedLS, lambda); err != nil {
		return SPD{}, err
	}

	return spd, nil
}
