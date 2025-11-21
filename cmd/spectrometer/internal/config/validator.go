package config

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"log/slog"

	"github.com/itohio/EasyRobot/types/spectrometer"
	"github.com/itohio/EasyRobot/x/marshaller/gocv"
)

// Validator validates configuration parameters.
type Validator struct {
	validateCameraID bool // Whether to validate camera ID via enumeration
}

// NewValidator creates a new config validator.
func NewValidator(validateCameraID bool) *Validator {
	return &Validator{
		validateCameraID: validateCameraID,
	}
}

// Validate validates the configuration and returns an error if invalid.
// Returns a multi-error containing all validation issues.
func (v *Validator) Validate(ctx context.Context, config *spectrometer.SpectrometerConfiguration) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	var errors []string

	// Validate source settings
	if err := v.validateSource(ctx, config.Source); err != nil {
		errors = append(errors, fmt.Sprintf("source: %v", err))
	}

	// Validate window (only for camera source)
	if config.Source != nil && config.Source.Type == spectrometer.SourceType_SOURCE_CAMERA {
		if err := v.validateWindow(config.Window, config.Source.GetCamera()); err != nil {
			errors = append(errors, fmt.Sprintf("window: %v", err))
		}
	}

	// Validate wavelength calibration
	if err := v.validateCalibration(config.Calibration); err != nil {
		errors = append(errors, fmt.Sprintf("calibration: %v", err))
	}

	// Validate wavelength range
	if err := v.validateWavelengthRange(config.Wavelength); err != nil {
		errors = append(errors, fmt.Sprintf("wavelength: %v", err))
	}

	// Validate signal processing
	if err := v.validateSignalProcessing(config.Processing); err != nil {
		errors = append(errors, fmt.Sprintf("processing: %v", err))
	}

	// Validate peak detection
	if err := v.validatePeakDetection(config.Peaks); err != nil {
		errors = append(errors, fmt.Sprintf("peaks: %v", err))
	}

	// Validate reference spectrum paths (warn only, don't fail)
	if err := v.validateReferencePaths(ctx, config.ReferenceSpectrums); err != nil {
		slog.Warn("Reference spectrum validation", "error", err)
		// Don't add to errors - this is a warning
	}

	if len(errors) > 0 {
		return fmt.Errorf("validation failed:\n  - %s", strings.Join(errors, "\n  - "))
	}

	return nil
}

// validateSource validates source settings.
func (v *Validator) validateSource(ctx context.Context, source *spectrometer.SourceSettings) error {
	if source == nil {
		return fmt.Errorf("source settings are required")
	}

	switch source.Type {
	case spectrometer.SourceType_SOURCE_CAMERA:
		camera := source.GetCamera()
		if camera == nil {
			return fmt.Errorf("camera settings are required when source type is camera")
		}
		return v.validateCamera(ctx, camera)
	case spectrometer.SourceType_SOURCE_SENSOR:
		sensor := source.GetSensor()
		if sensor == nil {
			return fmt.Errorf("sensor settings are required when source type is sensor")
		}
		return v.validateSensor(sensor)
	case spectrometer.SourceType_SOURCE_UNSPECIFIED:
		return fmt.Errorf("source type must be specified (camera or sensor)")
	default:
		return fmt.Errorf("unknown source type: %v", source.Type)
	}
}

// validateCamera validates camera settings.
func (v *Validator) validateCamera(ctx context.Context, camera *spectrometer.CameraSettings) error {
	if camera.Id == "" {
		return fmt.Errorf("camera ID is required")
	}

	// Validate camera ID exists (if enabled)
	if v.validateCameraID {
		if err := v.checkCameraIDExists(ctx, camera.Id); err != nil {
			return fmt.Errorf("camera ID validation: %w", err)
		}
	}

	// Validate resolution
	if camera.Width <= 0 {
		return fmt.Errorf("camera width must be positive, got %d", camera.Width)
	}
	if camera.Height <= 0 {
		return fmt.Errorf("camera height must be positive, got %d", camera.Height)
	}

	// Validate format (non-empty)
	if camera.Format == "" {
		return fmt.Errorf("camera format is required")
	}

	// Validate FPS
	if camera.Fps <= 0 {
		return fmt.Errorf("camera FPS must be positive, got %d", camera.Fps)
	}

	// Validate gain (if set)
	if camera.Gain != nil && *camera.Gain < 0 {
		return fmt.Errorf("camera gain must be non-negative, got %f", *camera.Gain)
	}

	// Validate exposure (if set)
	if camera.ExposureUs != nil && *camera.ExposureUs < 0 {
		return fmt.Errorf("camera exposure must be non-negative, got %d", *camera.ExposureUs)
	}

	return nil
}

// validateSensor validates sensor settings.
func (v *Validator) validateSensor(sensor *spectrometer.SensorSettings) error {
	if sensor.Type == "" {
		return fmt.Errorf("sensor type is required")
	}
	if sensor.DevicePath == "" {
		return fmt.Errorf("sensor device path is required")
	}
	if sensor.IntegrationTimeMs <= 0 {
		return fmt.Errorf("sensor integration time must be positive, got %d", sensor.IntegrationTimeMs)
	}
	if sensor.Gain < 0 {
		return fmt.Errorf("sensor gain must be non-negative, got %d", sensor.Gain)
	}
	return nil
}

// validateWindow validates window bounds.
func (v *Validator) validateWindow(window *spectrometer.SpectrumWindow, camera *spectrometer.CameraSettings) error {
	if window == nil {
		return nil // Window is optional
	}

	if camera == nil {
		return fmt.Errorf("camera settings required for window validation")
	}

	// Validate center point
	if window.Center == nil {
		return fmt.Errorf("window center is required")
	}
	if window.Center.X < 0 || window.Center.X >= camera.Width {
		return fmt.Errorf("window center X (%d) must be within image bounds [0, %d)", window.Center.X, camera.Width)
	}
	if window.Center.Y < 0 || window.Center.Y >= camera.Height {
		return fmt.Errorf("window center Y (%d) must be within image bounds [0, %d)", window.Center.Y, camera.Height)
	}

	// Validate size
	if window.Size == nil {
		return fmt.Errorf("window size is required")
	}
	if window.Size.Width <= 0 {
		return fmt.Errorf("window width must be positive, got %d", window.Size.Width)
	}
	if window.Size.Height <= 0 {
		return fmt.Errorf("window height must be positive, got %d", window.Size.Height)
	}

	// Validate window fits within image bounds
	halfWidth := window.Size.Width / 2
	halfHeight := window.Size.Height / 2
	if window.Center.X-halfWidth < 0 || window.Center.X+halfWidth > camera.Width {
		return fmt.Errorf("window X bounds [%d, %d) exceed image width [0, %d)",
			window.Center.X-halfWidth, window.Center.X+halfWidth, camera.Width)
	}
	if window.Center.Y-halfHeight < 0 || window.Center.Y+halfHeight > camera.Height {
		return fmt.Errorf("window Y bounds [%d, %d) exceed image height [0, %d)",
			window.Center.Y-halfHeight, window.Center.Y+halfHeight, camera.Height)
	}

	return nil
}

// validateCalibration validates wavelength calibration.
func (v *Validator) validateCalibration(calibration *spectrometer.WavelengthCalibration) error {
	if calibration == nil {
		return nil // Calibration is optional
	}

	// Validate polynomial
	if calibration.Polynomial == nil {
		return fmt.Errorf("calibration polynomial is required")
	}

	// Validate polynomial order matches number of points
	if calibration.Points != nil {
		numPoints := len(calibration.Points)
		if numPoints > 0 {
			// Polynomial should have at least as many coefficients as points for interpolation
			// For polynomial_2nd_order: need at least 3 coefficients (c0, c1, c2)
			// For polynomial_3rd_order: need at least 4 coefficients (c0, c1, c2, c3)
			coeffCount := len(calibration.Polynomial.Data)
			if calibration.Method == "polynomial_2nd_order" && coeffCount < 3 {
				return fmt.Errorf("polynomial_2nd_order requires at least 3 coefficients, got %d", coeffCount)
			}
			if calibration.Method == "polynomial_3rd_order" && coeffCount < 4 {
				return fmt.Errorf("polynomial_3rd_order requires at least 4 coefficients, got %d", coeffCount)
			}
		}
	}

	return nil
}

// validateWavelengthRange validates wavelength range.
func (v *Validator) validateWavelengthRange(wavelength *spectrometer.WavelengthRange) error {
	if wavelength == nil {
		return nil // Wavelength range is optional
	}

	// Typical range: 350-1000 nm
	if wavelength.Min < 0 {
		return fmt.Errorf("wavelength min must be non-negative, got %f", wavelength.Min)
	}
	if wavelength.Max <= wavelength.Min {
		return fmt.Errorf("wavelength max (%f) must be greater than min (%f)", wavelength.Max, wavelength.Min)
	}
	if wavelength.Max > 2000 {
		return fmt.Errorf("wavelength max (%f) is unusually high (typical range: 350-1000 nm)", wavelength.Max)
	}

	return nil
}

// validateSignalProcessing validates signal processing parameters.
func (v *Validator) validateSignalProcessing(processing *spectrometer.SignalProcessing) error {
	if processing == nil {
		return nil // Signal processing is optional
	}

	// Validate Savitzky-Golay filter
	if processing.Savgol != nil && processing.Savgol.Enabled {
		if err := v.validateSavitzkyGolay(processing.Savgol); err != nil {
			return fmt.Errorf("savgol: %w", err)
		}
	}

	// Validate row averaging
	if processing.RowAveraging != nil {
		if err := v.validateRowAveraging(processing.RowAveraging); err != nil {
			return fmt.Errorf("row_averaging: %w", err)
		}
	}

	return nil
}

// validateSavitzkyGolay validates Savitzky-Golay filter parameters.
func (v *Validator) validateSavitzkyGolay(savgol *spectrometer.SavitzkyGolayFilter) error {
	if savgol.WindowSize <= 0 {
		return fmt.Errorf("window_size must be positive, got %d", savgol.WindowSize)
	}
	if savgol.WindowSize%2 == 0 {
		return fmt.Errorf("window_size must be odd, got %d", savgol.WindowSize)
	}
	if savgol.PolynomialOrder < 0 {
		return fmt.Errorf("polynomial_order must be non-negative, got %d", savgol.PolynomialOrder)
	}
	if savgol.PolynomialOrder >= savgol.WindowSize {
		return fmt.Errorf("polynomial_order (%d) must be less than window_size (%d)", savgol.PolynomialOrder, savgol.WindowSize)
	}
	return nil
}

// validateRowAveraging validates row averaging parameters.
func (v *Validator) validateRowAveraging(avg *spectrometer.RowAveraging) error {
	if avg.Method == "" {
		return fmt.Errorf("row averaging method is required")
	}
	validMethods := []string{"mean", "median", "weighted_mean"}
	methodValid := false
	for _, m := range validMethods {
		if avg.Method == m {
			methodValid = true
			break
		}
	}
	if !methodValid {
		return fmt.Errorf("row averaging method must be one of %v, got %s", validMethods, avg.Method)
	}
	if avg.Rows <= 0 {
		return fmt.Errorf("row averaging rows must be positive, got %d", avg.Rows)
	}
	if avg.CenterRow != "" && avg.CenterRow != "middle" {
		// If center_row is specified and not "middle", try to parse as integer
		if _, err := strconv.Atoi(avg.CenterRow); err != nil {
			return fmt.Errorf("row averaging center_row must be 'middle' or a valid row index, got %s", avg.CenterRow)
		}
	}
	return nil
}

// validatePeakDetection validates peak detection parameters.
func (v *Validator) validatePeakDetection(peaks *spectrometer.PeakDetection) error {
	if peaks == nil {
		return nil // Peak detection is optional
	}

	if !peaks.Enabled {
		return nil // Skip validation if disabled
	}

	if peaks.Threshold < 0 || peaks.Threshold > 1 {
		return fmt.Errorf("peak detection threshold must be in range [0, 1], got %f", peaks.Threshold)
	}
	if peaks.MinDistance < 0 {
		return fmt.Errorf("peak detection min_distance must be non-negative, got %d", peaks.MinDistance)
	}

	return nil
}

// validateReferencePaths validates reference spectrum paths (warn only).
func (v *Validator) validateReferencePaths(ctx context.Context, entries []*spectrometer.ReferenceSpectrumEntry) error {
	if len(entries) == 0 {
		return nil
	}

	var warnings []string
	for _, entry := range entries {
		if entry == nil {
			continue
		}

		// Check if path is specified
		if path := entry.GetPath(); path != "" {
			// Resolve path (absolute or relative)
			resolvedPath := path
			if !filepath.IsAbs(path) {
				// For relative paths, we can't resolve without knowing the config file location
				// Just check if file exists at the given path
				resolvedPath = path
			}

			// Check if file exists
			if _, err := os.Stat(resolvedPath); os.IsNotExist(err) {
				warnings = append(warnings, fmt.Sprintf("reference spectrum '%s': path does not exist: %s", entry.Name, path))
			}
		}
	}

	if len(warnings) > 0 {
		return fmt.Errorf("reference spectrum path validation warnings: %s", strings.Join(warnings, "; "))
	}

	return nil
}

// checkCameraIDExists checks if camera ID exists by enumerating cameras.
func (v *Validator) checkCameraIDExists(ctx context.Context, cameraID string) error {
	unmarshaller := gocv.NewUnmarshaller()
	var devices []gocv.CameraInfo
	err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
	if err != nil {
		return fmt.Errorf("failed to enumerate cameras: %w", err)
	}

	// Try to parse camera ID as integer
	cameraIDInt, err := strconv.Atoi(cameraID)
	if err == nil {
		// Camera ID is numeric, check if it exists
		for _, dev := range devices {
			if dev.ID == cameraIDInt {
				return nil // Found
			}
		}
		return fmt.Errorf("camera ID %d not found in available cameras", cameraIDInt)
	}

	// Camera ID is string (path), check if it matches any camera path
	for _, dev := range devices {
		if dev.Path == cameraID || dev.Name == cameraID {
			return nil // Found
		}
	}

	return fmt.Errorf("camera ID '%s' not found in available cameras", cameraID)
}
