# Spectrometer Code Review - Separation of Concerns

## Executive Summary

After reviewing all spectrometer specs/plans against existing EasyRobot codebase capabilities, several functions planned for `spectrometer/internal/*` should be moved to or use existing implementations in:

1. **`x/math/colorscience`** - Already has peak detection, SPD calibration, CRI calculations
2. **`x/math/filter`** - Digital filters (Savitzky-Golay, Gaussian, Median should be added here)
3. **`x/marshaller/gocv`** - Should host GUI controls (trackbars, buttons, window management)

This document outlines the reorganization needed to honor single responsibility principle and avoid code duplication.

---

## Findings

### ✅ Already Implemented (Use, Don't Reimplement)

#### `x/math/colorscience` - Color Science Algorithms

**Peak Detection** ✅:
- `(spd SPD) Peaks(threshold, minProminence)` - Returns `[]Peak`
- `(spd SPD) Valleys(threshold, minProminence)` - Returns `[]Valley`
- `(spd SPD) DetectCalibrationPoints(referenceSPD, minConfidence)` - Auto-detection

**SPD Calibration** ✅:
- `(spd SPD) Calibrate(pairs ...float32)` - Wavelength calibration using cubic Catmull-Rom spline
- `(spd SPD) Reconstruct(sensorResponses, useDampedLS, lambda)` - Spectral reconstruction from sensor bands

**CRI Calculations** ✅:
- `(cs *ColorScience) ComputeCRI(spd SPD) (float32, error)` - Color Rendering Index (Ra, 0-100)

**Colorimetry** ✅:
- `ComputeXYZ(spd, wavelengths)` - XYZ tristimulus values
- `ComputeColorTemperature(spd)` - CCT and Duv
- `xyz.ToLAB(illuminant)` - LAB conversion
- `xyz.ToRGB(out255)` - RGB conversion

**Standard Illuminants** ✅:
- `LoadIlluminantSPD(name)` - D65, D50, A, etc.

**Spectral Reconstruction** ✅:
- `ReconstructSPDFromChannels(channels, targetWavelengths, useDampedLS, lambda)` - AS734x support
- `SensorChannelsToResponses(channels, targetWavelengths)` - Converts channels to responses

**Impact**: `spectrometer/internal/calibrate` should USE `colorscience.Peaks()` and `colorscience.SPD.Calibrate()`, not reimplement.

---

#### `x/math/filter` - Digital Filters

**Current Capabilities**:
- ✅ **Moving Average (MA)** - Already in `x/math/filter/ma` (`ma.MovingAverage`)
- FIR filters (`fir/` package)
- IIR filters (`iir/` package)
- Kalman filters (`kalman/` package)
- Other filters (AHRS, EK, SLAM)

**Missing (Should Be Added Here)**:
- ❌ **Savitzky-Golay Filter** - Should be in `x/math/filter`, not `spectrometer/internal/extract`
- ❌ **Gaussian Filter** - Missing from `x/math/filter`
- ❌ **Median Filter** - Missing from `x/math/filter`

**Impact**: `spectrometer/internal/extract/filtering.go` should USE `filter.SavitzkyGolay()` (to be implemented), not reimplement.

---

#### `x/marshaller/gocv` - Computer Vision & GUI

**Current Capabilities**:
- Image/video encoding/decoding
- Camera capture
- Display windows
- Event handling (keyboard, mouse)

**Missing (Should Be Added Here)**:
- ❌ **Trackbars (CreateTrackbar)** - For exposure, gain, filter parameters
- ❌ **Buttons (CreateButton)** - For capture dark/reference, toggles
- ❌ **Window management utilities** - Show/hide title, controls panel

**Impact**: `spectrometer/internal/controls` should USE `gocv.Trackbar` and `gocv.Button` (to be implemented in gocv marshaller), not reimplement.

---

### 🔧 Needs Enhancement (Document What's Missing)

#### `x/math/colorscience` Enhancements Needed

1. **Savitzky-Golay Filtering** (optional, should be in `x/math/filter` instead):
   - Currently missing from `x/math/filter/savgol`
   - Should be: `savgol.SavitzkyGolay(signal, windowSize, polynomialOrder)` - Apply SG filter
   - Parameters: window_size (odd), polynomial_order (< window_size - 1)
   - Use case: Smoothing while preserving peak shape

2. **Enhanced Peak Detection** (optional):
   - Current: `Peaks(threshold, minProminence)` - basic
   - Could add: Second-derivative method, CWT-based detection for noisy spectra

3. **CRI Enhancements** (optional):
   - Current: General CRI (Ra, 8 samples)
   - Could add: Extended CRI (R14, 14 samples), Individual Ri values

#### `x/math/filter` Enhancements Needed

1. **Savitzky-Golay Filter** (MUST ADD for spectrometer):
   - Package: `x/math/filter/savgol` (new subpackage)
   - Function: `SavitzkyGolay(signal, windowSize, polynomialOrder)` - Apply SG filter
   - Should work with `vec.Vector` and `SPD` types
   - Use case: Spectrum smoothing while preserving peak features
   - Algorithm: Industry-standard Savitzky-Golay convolution with polynomial fitting

2. **Gaussian Filter** (MUST ADD for spectrometer):
   - Package: `x/math/filter/gaussian` (new subpackage)
   - Function: `Gaussian(signal, sigma, windowSize)` - Apply Gaussian smoothing
   - Should work with `vec.Vector` and `SPD` types
   - Use case: Spectrum smoothing with Gaussian kernel

3. **Median Filter** (MUST ADD for spectrometer):
   - Package: `x/math/filter/median` (new subpackage)
   - Function: `Median(signal, windowSize)` - Apply median filter
   - Should work with `vec.Vector` and `SPD` types
   - Use case: Noise reduction while preserving edges/peaks

**Note**: Moving Average filter already exists in `x/math/filter/ma` ✅

---

### ❌ Should NOT Be in Spectrometer (Move to Shared Packages)

#### GUI Controls → `x/marshaller/gocv`

**Current Plan**: `spectrometer/internal/controls` - Camera/processing/display controls

**Should Be**: `x/marshaller/gocv` package additions:
- `Trackbar` interface and implementation
- `Button` interface and implementation
- `WindowManager` for show/hide title, controls panel

**Rationale**: GUI controls are reusable across all applications using gocv, not spectrometer-specific.

#### Peak Detection → Use `x/math/colorscience`

**Current Plan**: `spectrometer/internal/calibrate/peakdetect.go` - Peak detection

**Should Be**: Use `colorscience.SPD.Peaks()` directly

**Rationale**: Peak detection is a general spectral analysis operation, already implemented.

#### SPD Calibration → Use `x/math/colorscience`

**Current Plan**: `spectrometer/internal/calibrate/polynomial.go` - Polynomial fitting for calibration

**Should Be**: Use `colorscience.SPD.Calibrate()` directly

**Rationale**: Wavelength calibration is a general SPD operation, already implemented.

#### CRI Calculations → Use `x/math/colorscience`

**Current Plan**: Missing from plans!

**Should Be**: Add to plans - use `colorscience.ColorScience.ComputeCRI()`

**Rationale**: CRI is standard color science, already implemented.

---

## Required Changes

### 1. Command Structure Reorganization

**Current**:
```
cmd/spectrometer/
├── commands/
│   ├── measure/
│   │   ├── emissivity.go
│   │   ├── transmission.go
│   │   └── reflectance.go
│   ├── calibrate.go
│   ├── cameras.go
│   └── freerun.go
```

**Should Be**:
```
cmd/spectrometer/
├── calibrate/          # calibrate command (not commands/calibrate.go)
│   └── calibrate.go
├── measure/            # measure commands (not commands/measure/)
│   ├── emissivity.go
│   ├── transmission.go
│   ├── reflectance.go
│   ├── raman.go        # future
│   └── fluorescence.go # future
├── cameras/            # cameras command
│   └── cameras.go
├── freerun/            # freerun command
│   └── freerun.go
└── main.go             # CLI entry point, command routing
```

**Rationale**: Shorter paths, consistent with Go best practices, each command is a subdirectory.

---

### 2. Internal Module Reorganization

**Keep in `spectrometer/internal/*`** (spectrometer-specific orchestration):

- `internal/config/` - Config loading/saving for spectrometer (uses marshallers)
- `internal/extract/` - Spectrum extraction from camera pixels (spectrometer-specific)
  - **BUT**: Use `filter/savgol.SavitzkyGolay()`, `filter/gaussian.Gaussian()`, `filter/median.Median()`, `filter/ma.MovingAverage` for filtering (don't reimplement)
- `internal/calibrate/` - Calibration workflow orchestration
  - **BUT**: Use `colorscience.SPD.Peaks()` and `colorscience.SPD.Calibrate()` (don't reimplement peak detection/polynomial fitting)
- `internal/render/` - Spectrum visualization (spectrometer-specific rendering)
- `internal/correction/` - Correction workflow (dark, reference, masking)
- `internal/colorimetry/` - Colorimetry orchestration
  - **BUT**: Use `colorscience.ComputeXYZ()`, `colorscience.ComputeCRI()`, etc. (don't reimplement)
- `internal/measure/` - Measurement orchestration (single, averaged, transient)

**Move to `x/marshaller/gocv`**:

- ❌ `internal/controls/` → Move to `x/marshaller/gocv/controls/`
  - `camera.go` → `gocv/controls/trackbar.go` (generic trackbar implementation)
  - `processing.go` → Use trackbars from gocv
  - `display.go` → Use buttons/toggles from gocv

**Rationale**: GUI controls are application-agnostic and should be reusable.

---

### 3. Package Responsibilities

#### `spectrometer/*` (Commands - High-Level Orchestration)

**Responsibility**: Command-line interface, workflow orchestration, user interaction

**Should Do**:
- Parse command-line flags
- Load configuration
- Coordinate internal modules
- Handle command-specific workflows
- Error reporting to user

**Should NOT Do**:
- Implement DSP algorithms (use `x/math/dsp`)
- Implement color science algorithms (use `x/math/colorscience`)
- Implement GUI controls (use `x/marshaller/gocv`)

#### `spectrometer/internal/*` (Domain Orchestration)

**Responsibility**: Spectrometer-specific domain logic, workflow coordination

**Should Do**:
- Coordinate camera/sensor → extract/reconstruct → calibrate → correct → measure pipeline
- Manage spectrometer-specific state (window, calibration points, corrections)
- Orchestrate multiple algorithms from `x/math` packages

**Should NOT Do**:
- Reimplement peak detection (use `colorscience.SPD.Peaks()`)
- Reimplement calibration polynomials (use `colorscience.SPD.Calibrate()`)
- Reimplement filtering (use `filter/savgol.SavitzkyGolay()`, `filter/gaussian.Gaussian()`, `filter/median.Median()`, `filter/ma.MovingAverage`)
- Reimplement CRI (use `colorscience.ComputeCRI()`)

#### `x/math/colorscience` (Pure Color Science)

**Responsibility**: Color science algorithms, SPD operations, colorimetry calculations

**Already Has**:
- ✅ Peak/valley detection
- ✅ SPD wavelength calibration
- ✅ Spectral reconstruction
- ✅ XYZ, LAB, RGB conversions
- ✅ CCT calculations
- ✅ CRI calculations
- ✅ Standard illuminants

**Missing (Document for Future)**:
- Enhanced peak detection methods (optional)
- Extended CRI (R14, individual Ri values) (optional)

#### `x/math/filter` (Pure Signal Filtering)

**Responsibility**: Digital filters for signal processing

**Already Has**:
- ✅ Moving Average filter (`ma.MovingAverage`)
- ✅ FIR filters (`fir/` package)
- ✅ IIR filters (`iir/` package)
- ✅ Kalman filters (`kalman/` package)

**Missing (MUST ADD for Spectrometer)**:
- ❌ **Savitzky-Golay filter** (`savgol/` subpackage - critical for spectrometer)
- ❌ **Gaussian filter** (`gaussian/` subpackage - needed for spectrometer)
- ❌ **Median filter** (`median/` subpackage - needed for spectrometer)

**Note**: Filters belong in `x/math/filter`, not `x/math/dsp`. DSP package focuses on FFT, convolution, and window functions.

#### `x/math/dsp` (Pure Signal Processing)

**Responsibility**: Digital signal processing algorithms (FFT, convolution, window functions)

**Already Has**:
- ✅ FFT (1D/2D)
- ✅ Convolution
- ✅ Window functions
- ✅ Signal generation
- ✅ Measurements (RMS, SNR, Peak)

**Note**: Filters are NOT in DSP package - they belong in `x/math/filter`.

#### `x/marshaller/gocv` (GUI & Computer Vision)

**Responsibility**: GoCV integration, GUI controls, window management

**Already Has**:
- ✅ Image/video encoding/decoding
- ✅ Camera capture
- ✅ Display windows
- ✅ Event handling

**Missing (MUST ADD for Spectrometer)**:
- ❌ **Trackbars** (CreateTrackbar) - For exposure, gain, filter parameters
- ❌ **Buttons** (CreateButton) - For toggles, actions
- ❌ **Window utilities** - Show/hide title, controls panel management

---

## Updated Module Plans

### `spectrometer/internal/calibrate` - Updated Plan

**USE Instead of Implement**:
- Use `colorscience.SPD.Peaks(threshold, minProminence)` for peak detection
- Use `colorscience.SPD.Calibrate(pairs ...float32)` for wavelength calibration
- Use `colorscience.SPD.DetectCalibrationPoints(referenceSPD, minConfidence)` for auto-detection

**Implement Only**:
- Calibration targets database (Hg, Ar, Ne emission line wavelengths)
- Peak matching algorithm (match detected peaks to known wavelengths)
- Calibration workflow orchestration (user interaction, validation)
- Quality assessment (R² calculation - use `x/math/mat` for polynomial fitting)

**Rationale**: Peak detection and calibration are general SPD operations, not spectrometer-specific.

---

### `spectrometer/internal/extract` - Updated Plan

**USE Instead of Implement**:
- Use `filter.SavitzkyGolay()` for Savitzky-Golay filtering (when implemented in `x/math/filter/savgol`)
- Use `filter.Gaussian()` for Gaussian smoothing (when implemented in `x/math/filter/gaussian`)
- Use `filter.Median()` for median filtering (when implemented in `x/math/filter/median`)
- Use `ma.MovingAverage` for moving average (already exists in `x/math/filter/ma` ✅)
- Use `dsp.Windows` for window functions (if needed for DSP operations)

**Implement Only**:
- Window detection (OBB from image variance) - spectrometer-specific
- Row averaging (mean, median, weighted) - spectrometer-specific
- Window extraction from frames - spectrometer-specific

**Rationale**: Filtering is general signal processing, not spectrometer-specific. Filters belong in `x/math/filter`, not `x/math/dsp`.

---

### `spectrometer/internal/colorimetry` - Updated Plan

**USE Instead of Implement**:
- Use `colorscience.ColorScience.ComputeXYZ()` for XYZ
- Use `colorscience.ColorScience.ComputeColorTemperature()` for CCT
- Use `colorscience.ColorScience.ComputeCRI()` for CRI ✅
- Use `xyz.ToLAB()` for LAB conversion
- Use `xyz.ToRGB()` for RGB conversion
- Use `colorscience.LoadIlluminantSPD()` for standard illuminants

**Implement Only**:
- Colorimetry orchestration (coordinate multiple calculations)
- Display overlay rendering (spectrometer-specific visualization)

**Rationale**: Color science calculations are general, not spectrometer-specific.

---

### `spectrometer/internal/controls` - MOVE TO `x/marshaller/gocv`

**Current Plan**: `spectrometer/internal/controls` - GUI controls

**New Location**: `x/marshaller/gocv/controls/`

**New Structure**:
```
x/marshaller/gocv/
├── controls/
│   ├── trackbar.go      # Trackbar implementation (CreateTrackbar, callback handling)
│   ├── button.go        # Button implementation (CreateButton, callback handling)
│   └── window.go        # Window management utilities (show/hide title, controls panel)
├── ... (existing files)
```

**Usage in Spectrometer**:
```go
// In spectrometer/internal/colorimetry or spectrometer commands
import "github.com/itohio/EasyRobot/x/marshaller/gocv/controls"

// Create exposure trackbar
trackbar := controls.NewTrackbar("Exposure", "WindowName", 0, 100, 
    func(value int) {
        // Update camera exposure
        camera.SetExposure(value)
    })

// Create button for dark frame capture
button := controls.NewButton("Capture Dark", "WindowName",
    func() {
        // Capture dark frame
        darkFrame := captureDarkFrame()
    })
```

**Rationale**: GUI controls are reusable across all applications using gocv.

---

## Missing Features to Add to Plans

### CRI Calculations

**Status**: Already implemented in `x/math/colorscience`! ✅

**Action**: Add to `spectrometer/internal/colorimetry/PLAN.md`:
- [ ] Use `colorscience.ColorScience.ComputeCRI(spd)` for CRI calculations
- [ ] Display CRI in overlay (alongside CCT, XYZ/LAB)

**Rationale**: CRI is standard colorimetry, should be included in spectrometer plans.

---

## Required Enhancements to Shared Packages

### 1. `x/math/filter` - Add Missing Filters

**Priority**: HIGH (needed for spectrometer)

#### 1a. Savitzky-Golay Filter (`x/math/filter/savgol`)

**Implementation**:
```go
// In x/math/filter/savgol/savgol.go (new subpackage)

// SavitzkyGolay applies Savitzky-Golay smoothing filter to signal
// Parameters:
//   - signal: Input signal (vec.Vector)
//   - windowSize: Window size (must be odd, >= 3)
//   - polynomialOrder: Polynomial order (must be < windowSize - 1)
// Returns: Filtered signal (vec.Vector)
func SavitzkyGolay(signal vecTypes.Vector, windowSize, polynomialOrder int) (vecTypes.Vector, error)

// SavitzkyGolayFilter is a reusable filter for streaming applications
type SavitzkyGolayFilter struct {
    windowSize int
    polynomialOrder int
    // ... internal state (coefficients, buffer)
}

func NewSavitzkyGolayFilter(windowSize, polynomialOrder int) (*SavitzkyGolayFilter, error)

func (sg *SavitzkyGolayFilter) Process(sample float32) float32  // Single sample processing
func (sg *SavitzkyGolayFilter) ProcessBuffer(input, output vecTypes.Vector) error  // Buffer processing
func (sg *SavitzkyGolayFilter) Reset()  // Reset internal state
```

**Algorithm**: Industry-standard Savitzky-Golay (convolution with SG coefficients calculated via polynomial least-squares)

**Testing**: Unit tests with known signals, verify peak preservation

**Documentation**: Create `x/math/filter/savgol/SPEC.md`

#### 1b. Gaussian Filter (`x/math/filter/gaussian`)

**Implementation**:
```go
// In x/math/filter/gaussian/gaussian.go (new subpackage)

// Gaussian applies Gaussian smoothing filter to signal
// Parameters:
//   - signal: Input signal (vec.Vector)
//   - sigma: Standard deviation of Gaussian kernel
//   - windowSize: Window size (optional, auto-calculated from sigma if 0)
// Returns: Filtered signal (vec.Vector)
func Gaussian(signal vecTypes.Vector, sigma float32, windowSize int) (vecTypes.Vector, error)

// GaussianFilter is a reusable filter for streaming applications
type GaussianFilter struct {
    sigma float32
    windowSize int
    // ... internal state (kernel, buffer)
}

func NewGaussianFilter(sigma float32, windowSize int) (*GaussianFilter, error)

func (gf *GaussianFilter) Process(sample float32) float32
func (gf *GaussianFilter) ProcessBuffer(input, output vecTypes.Vector) error
func (gf *GaussianFilter) Reset()
```

**Testing**: Unit tests with known signals, verify smoothing

**Documentation**: Create `x/math/filter/gaussian/SPEC.md`

#### 1c. Median Filter (`x/math/filter/median`)

**Implementation**:
```go
// In x/math/filter/median/median.go (new subpackage)

// Median applies median filter to signal
// Parameters:
//   - signal: Input signal (vec.Vector)
//   - windowSize: Window size (must be odd, >= 3)
// Returns: Filtered signal (vec.Vector)
func Median(signal vecTypes.Vector, windowSize int) (vecTypes.Vector, error)

// MedianFilter is a reusable filter for streaming applications
type MedianFilter struct {
    windowSize int
    // ... internal state (buffer, sorted buffer for efficiency)
}

func NewMedianFilter(windowSize int) (*MedianFilter, error)

func (mf *MedianFilter) Process(sample float32) float32
func (mf *MedianFilter) ProcessBuffer(input, output vecTypes.Vector) error
func (mf *MedianFilter) Reset()
```

**Algorithm**: Efficient median calculation using partial sorting or heap

**Testing**: Unit tests with known signals, verify noise reduction while preserving edges

**Documentation**: Create `x/math/filter/median/SPEC.md`

---

### 2. `x/marshaller/gocv` - Add GUI Controls

**Priority**: HIGH (needed for spectrometer)

**Implementation**:
```go
// In x/marshaller/gocv/controls/trackbar.go

// Trackbar represents a GUI trackbar (slider)
type Trackbar struct {
    name string
    windowName string
    value int
    min, max int
    callback func(int)
}

// NewTrackbar creates a new trackbar in the specified window
func NewTrackbar(name, windowName string, min, max, initialValue int, 
    callback func(int)) (*Trackbar, error)

// UpdateValue updates trackbar value programmatically
func (tb *Trackbar) UpdateValue(value int) error

// GetValue returns current trackbar value
func (tb *Trackbar) GetValue() int

// Destroy removes trackbar from window
func (tb *Trackbar) Destroy() error
```

```go
// In x/marshaller/gocv/controls/button.go

// Button represents a GUI button
type Button struct {
    name string
    windowName string
    callback func()
}

// NewButton creates a new button in the specified window
func NewButton(name, windowName string, callback func()) (*Button, error)

// Destroy removes button from window
func (b *Button) Destroy() error
```

**Implementation Notes**:
- Use GoCV's `gocv.CreateTrackbar()` and `gocv.CreateButton()` (if available)
- Handle callbacks via GoCV's callback mechanism
- Manage lifecycle (creation, updates, destruction)

**Testing**: Unit tests with mock windows, integration tests with real display

**Documentation**: Add to `x/marshaller/gocv/SPEC.md`

---

## Updated File Structure

### Commands (High-Level Orchestration)

```
cmd/spectrometer/
├── main.go              # CLI entry point, command routing
├── SPEC.md              # This document
├── PLAN.md              # Implementation plan
├── REVIEW.md            # This review document
├── calibrate/           # calibrate command
│   └── calibrate.go
├── measure/             # measure commands
│   ├── emissivity.go
│   ├── transmission.go
│   ├── reflectance.go
│   ├── raman.go         # future
│   └── fluorescence.go  # future
├── cameras/             # cameras command
│   └── cameras.go
└── freerun/             # freerun command
    └── freerun.go
```

### Internal (Domain Orchestration)

```
cmd/spectrometer/internal/
├── config/              # Config management (uses marshallers)
├── extract/             # Spectrum extraction (uses filter/savgol, filter/gaussian, filter/median, filter/ma)
│   ├── extractor.go
│   ├── window.go        # Window detection (OBB from variance)
│   ├── averaging.go     # Row averaging
│   └── filtering.go     # Wrapper around filter functions
├── calibrate/           # Calibration orchestration (uses colorscience)
│   ├── calibrator.go    # Workflow orchestration
│   ├── targets.go       # Calibration targets (Hg, Ar, Ne)
│   ├── matcher.go       # Peak matching to targets
│   └── quality.go       # R² calculation (uses x/math/mat)
│   # NOTE: Peak detection uses colorscience.SPD.Peaks()
│   # NOTE: Polynomial fitting uses colorscience.SPD.Calibrate()
├── render/              # Visualization (spectrometer-specific)
│   ├── renderer.go
│   ├── waterfall.go
│   ├── graticule.go
│   ├── overlay.go
│   └── colors.go        # Wavelength-to-RGB (uses colorscience)
├── correction/          # Correction workflow
│   ├── dark.go
│   ├── reference.go
│   ├── mask.go
│   └── normalize.go
├── colorimetry/         # Colorimetry orchestration (uses colorscience)
│   ├── compute.go       # Wrapper around colorscience functions
│   ├── display.go       # Display overlay rendering
│   └── illuminants.go   # Wrapper around colorscience.LoadIlluminantSPD()
│   # NOTE: XYZ uses colorscience.ComputeXYZ()
│   # NOTE: CCT uses colorscience.ComputeColorTemperature()
│   # NOTE: CRI uses colorscience.ComputeCRI()
│   # NOTE: LAB uses xyz.ToLAB()
└── measure/             # Measurement orchestration
    ├── single.go
    ├── averaged.go
    ├── transient.go
    └── export.go
```

### Shared Packages (Enhancements Needed)

```
x/math/filter/
├── ... (existing files)
├── savgol/               # NEW: Savitzky-Golay filter
│   ├── savgol.go
│   ├── savgol_test.go
│   └── SPEC.md
├── gaussian/             # NEW: Gaussian filter
│   ├── gaussian.go
│   ├── gaussian_test.go
│   └── SPEC.md
├── median/               # NEW: Median filter
│   ├── median.go
│   ├── median_test.go
│   └── SPEC.md
├── ma/                   # EXISTING: Moving Average filter ✅
│   ├── ma.go
│   └── ma_test.go
└── SPEC.md               # UPDATE: Add new filters documentation

x/math/colorscience/
├── ... (existing files)
└── SPEC.md              # UPDATE: Document CRI, verify peak detection API

x/marshaller/gocv/
├── ... (existing files)
├── controls/            # NEW: GUI controls
│   ├── trackbar.go      # Trackbar implementation
│   ├── button.go        # Button implementation
│   └── window.go        # Window management utilities
└── SPEC.md              # UPDATE: Add GUI controls documentation
```

---

## Action Items

### Immediate (Update Specs)

1. ✅ Update `cmd/spectrometer/SPEC.md`:
   - Document use of `colorscience.SPD.Peaks()` for peak detection
   - Document use of `colorscience.SPD.Calibrate()` for wavelength calibration
   - Document use of `colorscience.ComputeCRI()` for CRI calculations
   - Move GUI controls to `x/marshaller/gocv` in dependencies
   - Update command structure (remove `commands/` directory)

2. ✅ Update `cmd/spectrometer/PLAN.md`:
   - Add CRI calculations to Phase 6
   - Document dependencies on `x/math/filter` for Savitzky-Golay, Gaussian, Median filters (when implemented)
   - Document dependencies on `x/marshaller/gocv/controls` for GUI
   - Update module plans to reflect "use, don't reimplement" approach

3. ✅ Update module SPEC/PLAN files:
   - `internal/calibrate/SPEC.md` - Document use of `colorscience` functions
   - `internal/calibrate/PLAN.md` - Remove peak detection implementation, add usage
   - `internal/extract/SPEC.md` - Document use of `filter/savgol.SavitzkyGolay()`, `filter/gaussian.Gaussian()`, `filter/median.Median()` (when available)
   - `internal/colorimetry/SPEC.md` - Document use of `colorscience` functions, add CRI
   - `internal/colorimetry/PLAN.md` - Add CRI to tasks
   - `internal/controls/SPEC.md` - Move to `x/marshaller/gocv/controls/SPEC.md`
   - `internal/controls/PLAN.md` - Move to `x/marshaller/gocv/controls/PLAN.md`

### Short-term (Package Enhancements)

4. ⚠️ **`x/math/filter`**: Implement missing filters
   - **`x/math/filter/savgol`**: Savitzky-Golay filter (HIGH priority, needed for spectrometer)
   - **`x/math/filter/gaussian`**: Gaussian filter (HIGH priority, needed for spectrometer)
   - **`x/math/filter/median`**: Median filter (HIGH priority, needed for spectrometer)
   - Timeline: Before spectrometer extract module implementation
   - Note: Moving Average filter already exists in `x/math/filter/ma` ✅

5. ⚠️ **`x/marshaller/gocv`**: Implement GUI controls (trackbars, buttons)
   - Priority: HIGH (needed for spectrometer)
   - Timeline: Before spectrometer controls integration

6. ✅ **`x/math/colorscience`**: Verify API completeness
   - CRI already implemented ✅
   - Peak detection already implemented ✅
   - SPD calibration already implemented ✅
   - No additional work needed

### Long-term (Optional Enhancements)

7. **`x/math/colorscience`**: Optional enhancements
   - Enhanced peak detection (CWT-based, second-derivative)
   - Extended CRI (R14, individual Ri values)
   - Savitzky-Golay in colorscience (or keep in DSP)

8. **`x/math/filter`**: Status
   - ✅ Moving Average filter already exists in `filter/ma`
   - ⚠️ Savitzky-Golay, Gaussian, Median filters need implementation (HIGH priority)

---

## Benefits of This Reorganization

1. **No Code Duplication**: Reuse existing implementations
2. **Better Testability**: Shared packages have comprehensive tests
3. **Reusability**: Other applications can use peak detection, CRI, etc.
4. **Maintainability**: Bug fixes in shared packages benefit all users
5. **Single Responsibility**: Each package has clear, focused responsibility
6. **Industry Standards**: Use proven algorithms from shared packages

---

## References

- `x/math/colorscience/SPEC.md` - Color science capabilities
- `x/math/filter/SPEC.md` - Filter capabilities
- `x/math/dsp/SPEC.md` - DSP capabilities (FFT, convolution, window functions)
- `x/marshaller/gocv/SPEC.md` - GoCV marshaller capabilities
- `x/marshaller/SPEC.md` - Marshaller subsystem overview

