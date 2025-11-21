# Calibrate Module - Implementation Plan

## Tasks

### 1. Calibration Targets
- [ ] Create built-in target database (Hg, Ar, Ne)
- [ ] Implement target loader/factory
- [ ] Support custom targets (user-provided wavelengths)
- [ ] Validate target wavelengths

### 2. Peak Detection
- [ ] **USE `colorscience.SPD.Peaks(threshold, minProminence)`** (already implemented)
- [ ] Create wrapper function for spectrometer configuration
- [ ] Convert spectrometer config (`types.spectrometer.PeakDetection`) to colorscience parameters
- [ ] Handle edge cases (spectrometer-specific validation)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 3. Peak Matching
- [ ] Implement greedy matching algorithm
- [ ] Implement wavelength order validation
- [ ] Implement distance validation
- [ ] Support user confirmation/editing
- [ ] Handle ambiguous matches

### 4. Wavelength Calibration
- [ ] **USE `colorscience.SPD.Calibrate(pairs ...float32)`** (already implemented)
- [ ] Create wrapper function for spectrometer calibration points
- [ ] Convert calibration points to variadic float32 pairs
- [ ] Extract polynomial coefficients from calibrated wavelengths (for R² calculation)
- [ ] Support 2nd and 3rd order polynomials (via colorscience implementation)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 4b. Polynomial Coefficients (for R² calculation)
- [ ] Extract polynomial coefficients from calibrated SPD wavelengths
- [ ] Use `x/math/mat` for polynomial fitting if needed for R² calculation
- [ ] Calculate R² quality metric

### 5. Quality Assessment
- [ ] Implement R² calculation
- [ ] Implement validation checks
- [ ] Add quality warnings/errors

### 6. Calibrator Integration
- [ ] Implement Calibrator interface
- [ ] Combine peak detection, matching, fitting
- [ ] Error handling and logging

### 7. Testing
- [ ] Unit tests for peak detection algorithms
- [ ] Unit tests for polynomial fitting
- [ ] Unit tests for peak matching
- [ ] Integration tests with known calibration data
- [ ] Test data generation (synthetic spectra with known peaks)

## Implementation Order

1. Calibration targets (simple data structures)
2. Polynomial fitting (uses x/math/mat, core functionality)
3. Peak detection (independent)
4. Peak matching (depends on peak detection)
5. Quality assessment (depends on fitting)
6. Calibrator integration (combines all)
7. Testing

## Research Needed

- Industry-standard peak detection algorithms for spectroscopy
- Best practices for polynomial fitting in spectroscopy
- Robust matching algorithms for calibration
- CWT (Continuous Wavelet Transform) peak detection libraries/algorithms

## Dependencies

- `x/math/mat` and `x/math/vec` must be available
- Extract module for intensity spectra (for testing)
- Config module for calibration settings

