# Calibrate Module - Specification

## Overview

The calibrate module orchestrates wavelength calibration workflow by coordinating peak detection (using `x/math/colorscience`), matching detected peaks to known calibration targets (emission line sources), and applying wavelength calibration (using `x/math/colorscience`). The module does NOT reimplement peak detection or polynomial fitting - it uses existing implementations from `x/math/colorscience`.

## Responsibilities

1. **Calibration Targets**: Provide built-in emission line databases (Hg, Ar, Ne, etc.)
2. **Peak Matching**: Match detected peaks (from `colorscience.SPD.Peaks()`) to known calibration wavelengths
3. **Wavelength Calibration**: Orchestrate `colorscience.SPD.Calibrate()` for wavelength mapping
4. **Quality Assessment**: Calculate R² and validate calibration quality (using `x/math/mat`)
5. **Workflow Orchestration**: User interaction, validation, error handling

## Interfaces

```go
// Calibrator performs wavelength calibration
type Calibrator interface {
    Calibrate(ctx context.Context, points []CalibrationPoint) (*types.spectrometer.WavelengthCalibration, error)
    Validate(ctx context.Context, cal *types.spectrometer.WavelengthCalibration) error
}

// PeakDetector wraps colorscience.SPD.Peaks() for spectrometer use
// Uses colorscience.SPD.Peaks(threshold, minProminence) internally
type PeakDetector interface {
    Detect(ctx context.Context, spd colorscience.SPD, config *types.spectrometer.PeakDetection) ([]colorscience.Peak, error)
}

// Note: Peak type is from colorscience.Peak:
//   type Peak struct {
//       Index      int       // Peak index in SPD
//       Wavelength float32   // Peak wavelength
//       Value      float32   // Peak intensity
//       Prominence float32   // Peak prominence
//   }

// TargetMatcher matches detected peaks to calibration targets
type TargetMatcher interface {
    Match(ctx context.Context, peaks []Peak, target *CalibrationTarget) ([]CalibrationPoint, error)
}

// Calibrator wraps colorscience.SPD.Calibrate() for spectrometer use
// Uses colorscience.SPD.Calibrate(pairs ...float32) internally
type Calibrator interface {
    Calibrate(ctx context.Context, spd colorscience.SPD, points []CalibrationPoint) (colorscience.SPD, *types.spectrometer.WavelengthCalibration, error)
    Validate(ctx context.Context, cal *types.spectrometer.WavelengthCalibration) error
}

// CalibrationTarget represents an emission line source
type CalibrationTarget struct {
    Name        string      // Name (e.g., "hg", "ar", "neon")
    Wavelengths []float64   // Known emission wavelengths (nm)
    Description string      // Description
}
```

## Peak Detection

**Uses `colorscience.SPD.Peaks(threshold, minProminence)`**:

```go
// Detect peaks in spectrum
peaks := spd.Peaks(0.5, 0.1) // threshold=0.5, minProminence=0.1
// Returns []colorscience.Peak sorted by wavelength
// Each peak has: Index, Wavelength, Value, Prominence
```

**Implementation**: The `colorscience.SPD.Peaks()` method already implements:
- Local maxima detection
- Threshold filtering (relative threshold)
- Prominence filtering (minimum prominence)
- Edge case handling

**Spectrometer-Specific**: The calibrate module wraps this function for spectrometer configuration parameters (from `types.spectrometer.PeakDetection` config).

**Note**: Peak detection is a general SPD operation, not spectrometer-specific. The implementation exists in `x/math/colorscience` and should be used, not reimplemented.

## Calibration Targets

**Built-in Targets**:
- **Hg (Mercury)**: 405.4, 436.6, 546.1, 577.0, 579.1, 623.4 nm
- **Ar (Argon)**: 415.9, 427.2, 451.1, 459.0, 514.5 nm
- **Ne (Neon)**: 540.1, 585.2, 588.2, 594.5, 603.0, 616.4 nm
- **Custom**: User-provided wavelength list

**Target Selection**:
- Load target by name from built-in database
- Support custom targets (list of wavelengths)
- Validate target wavelengths (reasonable ranges)

## Peak Matching

**Algorithm**:
1. Sort detected peaks by intensity (descending)
2. Sort calibration target wavelengths
3. Attempt greedy matching:
   - Match strongest detected peak to nearest target wavelength
   - Validate match (wavelength order preserved, reasonable distance)
   - Continue with remaining peaks
4. User confirmation/editing interface

**Validation**:
- Wavelength order must be monotonic (no inversions)
- Matched distances must be reasonable (not too far from expected)
- Minimum number of matches required (default: 3)

## Wavelength Calibration

**Uses `colorscience.SPD.Calibrate(pairs ...float32)`**:

```go
// Calibrate SPD using known (index, wavelength) pairs
calibrated, err := spd.Calibrate(
    0, 400.0,  // Index 0 = 400nm
    400, 565.0, // Index 400 = 565nm
    800, 750.0, // Index 800 = 750nm
)
// Uses cubic Catmull-Rom spline interpolation between calibration points
// Linear extrapolation before first and after last calibration point
```

**Implementation**: The `colorscience.SPD.Calibrate()` method already implements:
- Cubic Catmull-Rom spline interpolation between calibration points
- Linear extrapolation for indices outside calibration range
- Validation (at least 2 points, no duplicate indices, index range checking)

**Polynomial Coefficients** (for R² calculation):
- Extract polynomial coefficients from calibrated wavelengths for R² calculation
- Use `x/math/mat` for polynomial fitting if needed for quality metrics

**Spectrometer-Specific**: The calibrate module wraps this function and calculates R² quality metric using `x/math/mat`.

**Note**: Wavelength calibration is a general SPD operation, not spectrometer-specific. The implementation exists in `x/math/colorscience` and should be used, not reimplemented.

## Quality Assessment

**R² Calculation**:
- Compare predicted vs. actual wavelengths at calibration points
- Calculate correlation coefficient
- R² = corr²
- Target: R² > 0.999 for 4+ points

**Validation**:
- Minimum 3 calibration points required
- R² must be above threshold (configurable, default: 0.99)
- Warn if R² is low (possible misidentification)

## Dependencies

- `x/math/mat` - Matrix operations, polynomial fitting
- `x/math/vec` - Vector operations
- `types/spectrometer` - Configuration types
- `log/slog` - Structured logging

## Testing

- Unit tests for peak detection (synthetic spectra)
- Unit tests for polynomial fitting
- Unit tests for peak matching
- Integration tests with real calibration data
- Test data generation (synthetic spectra with known peaks)

