# Spectrometer Application - Implementation Plan

## Overview

This document tracks the high-level implementation progress for the spectrometer application. The application is divided into modular components, each with its own SPEC.md and PLAN.md for detailed tracking.

## Implementation Phases

**IMPORTANT**: Implementation starts with **Phase 0: CR30 Device Support** as the first phase. This provides immediate value with a working spectrometer and establishes the framework for future device support.

### Phase 0: CR30 Device Support & Framework (Weeks 1-2) ⭐ **FIRST PHASE**

**Goal**: Implement CR30 device support with spectrum measurement, rendering, and patch comparison capabilities.

**Rationale**: CR30 is a complete, working spectrometer device that provides direct spectrum measurements. Starting here:
- Provides immediate working spectrometer functionality
- Establishes framework for device "obtainers" (extensible to other devices)
- Implements spectrum renderer (reusable for all measurement modes)
- Validates colorimetry calculations end-to-end
- Provides foundation for camera-based spectrometer (Phase 2+)

**Tasks**:

- [ ] **Obtainer Framework** (`internal/obtainer/`)
  - [ ] Define `Obtainer` interface (devices that can obtain spectra)
    ```go
    type Obtainer interface {
        Connect(ctx context.Context) error
        Disconnect() error
        Measure(ctx context.Context) (colorscience.SPD, error)
        Wavelengths() vec.Vector
        DeviceInfo() DeviceInfo
    }
    ```
  - [ ] Device registry (register obtainers: `cr30`, future: `as734x`, etc.)
  - [ ] Obtainer factory (create obtainers from device type string)
  - [ ] CR30 obtainer implementation (wraps `x/devices/cr30`)

- [ ] **Spectrum Renderer** (`internal/render/`)
  - [ ] Core spectrum plot rendering (wavelength vs. intensity)
  - [ ] Colorimetry overlay rendering (XYZ, LAB, RGB HEX)
    - Rotate display on spacebar: XYZ → LAB → RGB HEX → repeat
  - [ ] Color swatch rendering (RGB color representation)
  - [ ] Patch comparison rendering (half reference, half measured)
  - [ ] Delta E (ΔE) calculation and display (CIE76 formula)

- [ ] **Patch Loading** (`internal/config/` or new `internal/patches/`)
  - [ ] YAML/JSON patch file parser
  - [ ] Patch structure (name, XYZ, LAB, RGB reference values)
  - [ ] Patch validation

- [ ] **Measure Command** (`measure/measure.go`)
  - [ ] Device selection via `-device` flag
  - [ ] CR30 device initialization (`--port`, `--baud` flags)
  - [ ] Measurement workflow:
    - Measure spectrum from device
    - Calculate XYZ, LAB, RGB HEX using `colorscience`
    - Display spectrum graph with colorimetry overlay
  - [ ] Multiple readings support (`--readings=N`):
    - Perform N measurements
    - Save N spectrums with N XYZ/LAB values
  - [ ] Patch-based measurement (`--patches=path`):
    - Load patches from YAML/JSON
    - Display patches
    - For each patch: measure, compare, display delta E
  - [ ] Export functionality (CSV, JSON, Proto for measurements)

**Key Features**:

1. **Without `-patches` flag**:
   ```
   spectrometer measure -device=cr30 --port=/dev/ttyUSB0 --readings=5
   ```
   - Measures 5 spectrums
   - Displays each spectrum with XYZ, LAB, RGB HEX
   - Shows RGB color swatch
   - Saves 5 spectrums with 5 XYZ/LAB values

2. **With `-patches` flag**:
   ```
   spectrometer measure -device=cr30 --port=/dev/ttyUSB0 --patches=patches.yaml
   ```
   - Loads patches from `patches.yaml`
   - Displays patches as color swatches
   - For each patch:
     - Measures spectrum
     - Calculates XYZ, LAB, RGB HEX
     - Calculates delta E (ΔE) vs. reference patch
     - Displays patch as half reference (left) and half measured (right)
     - Displays RGB color, XYZ/LAB/RGB HEX (rotate on spacebar), delta E

**Dependencies**:
- `x/devices/cr30` - CR30 device communication ✅ (already implemented)
- `x/math/colorscience` - XYZ, LAB, RGB calculations ✅ (already implemented)
- `cmd/display/destination` - Display window ✅ (already implemented)
- `x/marshaller/yaml` - YAML patch file loading ✅ (already implemented)
- `x/marshaller/json` - JSON patch file loading ✅ (already implemented)

**Deliverables**:
- `spectrometer measure -device=cr30` command works
- Can measure N readings and save N spectrums with N XYZ/LAB values
- Can load patches from YAML/JSON and compare measurements
- Spectrum renderer displays spectrum, colorimetry, patches, and delta E
- Framework extensible to other devices (e.g., `as734x`)

---

### Phase 1: Foundation & Configuration (Weeks 3-4)
**Goal**: Core infrastructure and configuration (moved after CR30 for immediate value)

- [ ] **Proto Definitions** (if not already done in Phase 0)
  - [ ] Create `proto/types/spectrometer/config.proto` ✅ (already created)
  - [ ] Create `proto/types/spectrometer/measurement.proto` ✅ (already created)
  - [ ] Build proto files (`make proto`)
  - [ ] Verify generated Go code

- [ ] **Config Module** (`internal/config/`)
  - [ ] Config structure and validation
  - [ ] Config loader (proto/YAML/JSON via marshallers)
  - [ ] Config saver (proto/YAML/JSON via marshallers)
  - [ ] Reference spectrum loading (resolve paths, load matrices)
  - [ ] Config validation and error handling

- [ ] **Main Entry Point** (`main.go`)
  - [ ] CLI framework setup (flag parsing)
  - [ ] Command routing structure
  - [ ] Logging setup (slog with -v=N and -vv flags)
  - [ ] Integration with display/source and display/destination

- [ ] **Cameras Command** (`cameras/cameras.go`)
  - [ ] Reuse `source.ListCameras()`
  - [ ] Format and display camera information
  - [ ] Exit after listing

**Dependencies**: Phase 0 (CR30 framework established)

**Deliverables**:
- Working proto definitions
- Config can be loaded/saved
- `spectrometer cameras` command works
- Logging infrastructure in place

---

### Phase 2: Core Extraction and Calibration (Weeks 5-7)
**Goal**: Extract spectrum from images and calibrate wavelength mapping

- [ ] **Extract Module** (`internal/extract/`)
  - [ ] Window detection (OBB from image variance analysis)
  - [ ] Row averaging algorithms (mean, median, weighted)
  - [ ] Use `filter.SavitzkyGolay()` for filtering (when implemented in `x/math/filter/savgol`)
  - [ ] Use `filter.Gaussian()` for Gaussian smoothing (when implemented in `x/math/filter/gaussian`)
  - [ ] Use `filter.Median()` for median filtering (when implemented in `x/math/filter/median`)
  - [ ] Use `ma.MovingAverage` for moving average (already exists ✅)
  - [ ] Spectrum extraction from frames

**Dependencies**: `x/math/filter` must implement Savitzky-Golay, Gaussian, and Median filters first

- [ ] **Calibrate Module** (`internal/calibrate/`)
  - [ ] Use `colorscience.SPD.Peaks()` for peak detection (already implemented)
  - [ ] Calibration targets (Hg, Ar, Ne emission lines)
  - [ ] Peak matching with targets
  - [ ] Use `colorscience.SPD.Calibrate()` for wavelength calibration (already implemented)
  - [ ] R² calculation and validation (uses `x/math/mat` for polynomial fitting)

- [ ] **Calibrate Command** (`calibrate/calibrate.go`)
  - [ ] Interactive window detection
  - [ ] Interactive peak selection
  - [ ] Calibration workflow
  - [ ] Config export (stdout/file)

**Dependencies**: Phase 1 (config, proto)

**Deliverables**:
- Spectrum can be extracted from camera frames
- Window can be detected automatically
- Calibration can be performed interactively
- `spectrometer calibrate camera` command works

---

### Phase 3: Correction and Processing (Week 8)
**Goal**: Spectrum correction and signal processing

- [ ] **Correction Module** (`internal/correction/`)
  - [ ] Dark frame capture and subtraction
  - [ ] Reference spectrum loading and application
  - [ ] Spectrum masking (region-of-interest)
  - [ ] Normalization and linearization

- [ ] **Signal Processing**
  - [ ] Camera linearization (polynomial correction)
  - [ ] Integration with extract module filtering

**Dependencies**: Phase 2 (extraction)

**Deliverables**:
- Dark frame correction works
- Reference spectrum subtraction/normalization works
- Spectrum masking works

---

### Phase 4: Visualization (Week 9)
**Goal**: Render spectrum graphs and displays

- [ ] **Render Module** (`internal/render/`)
  - [ ] Spectrum graph rendering (wavelength-to-RGB)
  - [ ] Waterfall display rendering
  - [ ] Graticule generation and rendering
  - [ ] Overlay rendering (illuminants, calibration lines)
  - [ ] Colorimetry display overlays

- [ ] **GUI Controls** (via `x/marshaller/gocv/controls/`)
  - [ ] Use `gocv/controls.Trackbar` for camera controls (exposure, gain) - to be implemented in gocv
  - [ ] Use `gocv/controls.Trackbar` for processing controls (filter parameters) - to be implemented in gocv
  - [ ] Use `gocv/controls.Button` for display toggles (CCT, XYZ/LAB, overlays) - to be implemented in gocv
  - [ ] Event handling (keyboard, mouse) - via gocv display

**Dependencies**: `x/marshaller/gocv` must implement `gocv/controls/` package first

**Dependencies**: Phase 2 (calibration), Phase 3 (correction)

**Deliverables**:
- Spectrum graphs can be rendered
- Waterfall display works
- GUI controls functional (trackbars, buttons)

---

### Phase 5: Basic Measurement (Week 10)
**Goal**: Emissivity measurements working

- [ ] **Measure Module - Base** (`internal/measure/`)
  - [ ] Single-shot measurement
  - [ ] Averaged measurement (frame averaging)
  - [ ] Transient measurement (time-series)
  - [ ] Export via marshallers (CSV, JSON, proto)

- [ ] **Measure Emissivity Command** (`measure/emissivity.go`)
  - [ ] Integration with extract, correction, render modules
  - [ ] Real-time display via display/destination
  - [ ] Data export

- [ ] **Freerun Command** (`freerun/freerun.go`)
  - [ ] Simple real-time display
  - [ ] Basic interactivity
  - [ ] No data logging

**Dependencies**: Phase 2, 3, 4

**Deliverables**:
- `spectrometer measure emissivity` command works
- `spectrometer freerun` command works
- Basic measurements can be exported

---

### Phase 6: Colorimetry and Advanced Features (Week 11)
**Goal**: Colorimetry calculations and advanced display features

- [ ] **Colorimetry Module** (`internal/colorimetry/`)
  - [ ] Use `colorscience.ComputeXYZ()` for XYZ calculations
  - [ ] Use `colorscience.ComputeColorTemperature()` for CCT
  - [ ] Use `colorscience.ComputeCRI()` for CRI calculations ✅
  - [ ] Use `xyz.ToLAB()` for LAB conversions
  - [ ] Use `xyz.ToRGB()` for RGB conversions
  - [ ] Use `colorscience.LoadIlluminantSPD()` for standard illuminants
  - [ ] Colorimetry orchestration (coordinate multiple calculations)
  - [ ] Display overlay rendering (spectrometer-specific visualization)

- [ ] **Advanced Display Features**
  - [ ] CCT display toggle
  - [ ] XYZ/LAB display overlay
  - [ ] CRI display (Ra value) ✅
  - [ ] Illuminant overlays
  - [ ] Calibration line overlays

**Dependencies**: Phase 4 (render), `x/math/colorscience` (already has CRI ✅)

**Deliverables**:
- Colorimetry calculations work (XYZ, CCT, CRI, LAB, RGB)
- Advanced display overlays functional
- CRI calculations integrated ✅

---

### Phase 7: Transmission and Reflectance (Week 12)
**Goal**: Illuminant-based measurements

- [ ] **Measure Transmission Command** (`measure/transmission.go`)
  - [ ] Illuminant selection (D65, R, G, B, RGB, UV)
  - [ ] Reference spectrum capture
  - [ ] Transmission calculation (sample/reference)
  - [ ] Workflow guidance

- [ ] **Measure Reflectance Command** (`measure/reflectance.go`)
  - [ ] D65 illuminant (primary)
  - [ ] White reference capture
  - [ ] Reflectance calculation
  - [ ] Workflow guidance

**Dependencies**: Phase 5 (measure base), Phase 6 (colorimetry)

**Note**: Hardware illuminant control marked as "to be implemented" - software supports it but hardware integration comes later.

**Deliverables**:
- `spectrometer measure transmission` command works
- `spectrometer measure reflectance` command works
- Illuminant selection framework in place

---

### Phase 8: Advanced Measurement Modes (Weeks 11-12)
**Goal**: Raman and fluorescence measurements

- [ ] **Measure Raman Command** (`measure/raman.go`)
  - [ ] RGB LED illuminant selection (individual/single)
  - [ ] Background spectrum capture
  - [ ] Raman signal calculation
  - [ ] Separate vs. combined RGB handling

- [ ] **Measure Fluorescence Command** (`measure/fluorescence.go`)
  - [ ] UV illuminant flash mode
  - [ ] Decay time-series capture
  - [ ] Exponential decay fitting
  - [ ] Lifetime constant calculation (τ)

**Dependencies**: Phase 7 (illuminant framework)

**Deliverables**:
- `spectrometer measure raman` command works
- `spectrometer measure fluorescence` command works
- Decay analysis functional

---

### Phase 9: CSV Marshaller (Week 13)
**Goal**: CSV export support

- [ ] **CSV Marshaller** (`x/marshaller/csv/`)
  - [ ] Vector marshalling/unmarshalling
  - [ ] Matrix marshalling/unmarshalling
  - [ ] Tensor marshalling/unmarshalling (1D/2D only)
  - [ ] `WithHeader(true/false)` option
  - [ ] `WithZeroRowHeader(true/false)` option
  - [ ] Integration with measure export

**Dependencies**: Phase 5 (measure export framework)

**Deliverables**:
- CSV marshaller implemented and tested
- All measurement commands can export to CSV

---

### Phase 10: Polish and Testing (Week 14)
**Goal**: Testing, documentation, error handling

- [ ] **Testing**
  - [ ] Unit tests for all modules (using testify)
  - [ ] Integration tests for commands
  - [ ] Test data generation (synthetic spectra)
  - [ ] Edge case handling

- [ ] **Error Handling**
  - [ ] Comprehensive error wrapping
  - [ ] User-friendly error messages
  - [ ] Graceful degradation

- [ ] **Documentation**
  - [ ] README.md with usage examples
  - [ ] Module documentation
  - [ ] Algorithm documentation
  - [ ] Calibration guide

- [ ] **Performance Optimization**
  - [ ] Profiling and optimization
  - [ ] Memory management
  - [ ] Frame processing efficiency

**Dependencies**: All previous phases

**Deliverables**:
- Comprehensive test coverage
- Production-ready error handling
- Complete documentation
- Optimized performance

---

## Module Breakdown

### Core Modules

1. **config/** - Configuration management
   - Load/save via marshallers
   - Reference spectrum path resolution
   - Validation

2. **extract/** - Spectrum extraction
   - Window detection (industry standard algorithms)
   - Row averaging
   - Filtering (Savitzky-Golay)

3. **calibrate/** - Wavelength calibration
   - Peak detection (robust algorithms)
   - Polynomial fitting
   - Target matching

4. **correction/** - Spectrum correction
   - Dark frame handling
   - Reference spectrum application
   - Masking

5. **render/** - Visualization
   - Graph rendering
   - Waterfall display
   - Overlays

6. **controls/** - GUI controls
   - Trackbars (gocv)
   - Event handling

7. **colorimetry/** - Colorimetry calculations
   - CCT, XYZ, LAB
   - Illuminant SPDs

8. **measure/** - Measurement modes
   - Single, averaged, transient
   - Export

### Commands

**Command Structure** (updated - no `commands/` subdirectory):

1. **cameras/** - List cameras (Phase 1)
2. **calibrate/** - Calibration workflow (Phase 2)
3. **measure/** - Measurement commands (Phase 5, 7, 8)
   - `emissivity.go` - Emissivity measurements (Phase 5)
   - `transmission.go` - Transmission measurements (Phase 7)
   - `reflectance.go` - Reflectance measurements (Phase 7)
   - `raman.go` - Raman spectroscopy (Phase 8)
   - `fluorescence.go` - Fluorescence/luminescence (Phase 8)
4. **freerun/** - Real-time display (Phase 5)

---

## Development Standards

### Go Version
- **Go 1.25** (assumed latest features)

### Logging
- **Package**: `log/slog`
- **Flags**: `-v=N` (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)
- **Flag**: `-vv` (shortcut for `-v=4`, TRACE level)
- Structured logging with context

### Testing
- **Package**: `testify`
- Unit tests for all modules
- Integration tests for commands
- Test data generation utilities

### Code Quality
- SOLID principles
- Go best practices (Effective Go, Code Review Comments)
- Function length limits (30 lines, extract when makes sense)
- Package by feature/domain
- Error handling (never ignore, wrap with context)
- No naked returns (except very short functions)

### Algorithms
- Use industry standard algorithms (not verbatim PySpectrometer2)
- Leverage gocv for computer vision operations
- Use `x/math/mat` and `x/math/vec` for numerical operations
- Prefer standard library over custom implementations

---

## Dependencies on External Packages

### Required Implementations
- **CSV Marshaller** (`x/marshaller/csv`) - Phase 9
- **Hardware Illuminant Control** (`x/devices/spectrometer`) - Future (for Phase 7+)

### Existing Packages (Reuse)

**Core Algorithms** (use, don't reimplement):
- `x/math/colorscience` - Colorimetry, peak detection, SPD calibration, CRI ✅
  - `SPD.Peaks()` - Peak detection
  - `SPD.Calibrate()` - Wavelength calibration
  - `SPD.DetectCalibrationPoints()` - Auto-calibration point detection
  - `ColorScience.ComputeXYZ()` - XYZ calculations
  - `ColorScience.ComputeColorTemperature()` - CCT calculations
  - `ColorScience.ComputeCRI()` - CRI calculations ✅
  - `xyz.ToLAB()` - LAB conversions
  - `xyz.ToRGB()` - RGB conversions
  - `LoadIlluminantSPD()` - Standard illuminants
  - `SPD.Reconstruct()` - Spectral reconstruction from sensor bands

**Signal Processing** (use, don't reimplement):
- `x/math/filter` - Digital filters
  - `ma.MovingAverage` - Moving average filter (**already exists** ✅)
  - `savgol.SavitzkyGolay()` - Savitzky-Golay filter (**needs implementation in `filter/savgol`**)
  - `gaussian.Gaussian()` - Gaussian smoothing (**needs implementation in `filter/gaussian`**)
  - `median.Median()` - Median filter (**needs implementation in `filter/median`**)
  - `fir` - FIR filters (already exists)
  - `iir` - IIR filters (already exists)
- `x/math/dsp` - DSP algorithms (FFT, convolution, window functions)
  - Window functions
  - Signal generation
  - Convolution operations

**Infrastructure**:
- `cmd/display/source` - Camera capture
- `cmd/display/destination` - Display/window management
- `x/math/mat` - Matrix operations (polynomial fitting, R² calculation)
- `x/math/vec` - Vector operations
- `x/marshaller/gocv` - Image/camera handling, GUI controls (**to be implemented in `gocv/controls/`**)
- `x/marshaller/proto` - Protobuf I/O
- `x/marshaller/yaml` - YAML I/O
- `x/marshaller/json` - JSON I/O

---

## Risk Mitigation

### Algorithm Implementation
- **Risk**: PySpectrometer2 algorithms may not be industry standard
- **Mitigation**: Research industry-standard algorithms, use gocv built-in methods where available, validate against reference implementations

### Hardware Integration
- **Risk**: Illuminant hardware control not yet implemented
- **Mitigation**: Design software API first, mark hardware integration as future work, support manual illuminant control for testing

### Performance
- **Risk**: Real-time processing may be slow
- **Mitigation**: Profile early, optimize critical paths, use goroutines for parallel processing

### Testing
- **Risk**: Lack of test data for spectra
- **Mitigation**: Generate synthetic test data, create test utilities, document test data format

---

## Success Criteria

1. **Functional**
   - All commands work as specified
   - Configuration can be saved/loaded
   - Measurements export correctly

2. **Quality**
   - >80% test coverage
   - All linter errors resolved
   - Comprehensive error handling

3. **Performance**
   - Real-time display at ≥30 FPS (camera-limited)
   - Measurement export < 1 second

4. **Usability**
   - Clear error messages
   - Helpful CLI help text
   - Intuitive workflows

---

## Progress Tracking

Update this section as work progresses:

- [ ] **Phase 0: CR30 Device Support & Framework** ⭐ **START HERE**
- [ ] Phase 1: Foundation & Configuration
- [ ] Phase 2: Core Extraction and Calibration
- [ ] Phase 3: Correction and Processing
- [ ] Phase 4: Visualization
- [ ] Phase 5: Basic Measurement
- [ ] Phase 6: Colorimetry and Advanced Features
- [ ] Phase 7: Transmission and Reflectance
- [ ] Phase 8: Advanced Measurement Modes
- [ ] Phase 9: CSV Marshaller & Final Polish
- [ ] Phase 10: Hardware Illuminant Control (Future)

