# Spectrometer Application - Specification

## Overview

The spectrometer application is a professional-grade scientific instrument for capturing, analyzing, and displaying spectral data from diffraction grating spectrometers. It processes images captured from a webcam, extracts spectrum data from a defined window, applies calibration, and outputs wavelength-calibrated Spectral Power Distributions (SPDs).

The application is built as a modular Go command-line tool following SOLID principles, leveraging EasyRobot's `mat`, `vec`, `colorimetry`, and `marshaller` packages for robust scientific computation.

**Key Capabilities:**
- Real-time spectrum capture and display
- Interactive calibration using known spectral lines
- Precise wavelength mapping via polynomial fitting
- Multiple measurement modes (single, averaged, transient)
- Waterfall visualization for time-series analysis
- Professional data export formats
- GUI controls for camera settings (exposure, gain) using gocv trackbars/buttons
- Dark frame and reference spectrum capture/correction
- Spectrum masking (region-of-interest filtering)
- Colorimetry calculations (CCT, XYZ, LAB) with display toggles
- Overlay support for standard illuminants and calibration lines

**Design Philosophy:**
- Modular architecture with clear separation of concerns
- Reusable components compatible with EasyRobot ecosystem
- Command-based interface following Unix philosophy
- Configuration-driven operation for reproducibility

---

## Architecture

### High-Level Structure

```
cmd/spectrometer/
├── main.go                    # CLI entry point, command routing
├── SPEC.md                    # This document
├── PLAN.md                    # Implementation plan
├── REVIEW.md                  # Code review and separation of concerns
├── calibrate/                 # calibrate command
│   └── calibrate.go
├── measure/                   # measure commands
│   ├── emissivity.go         # measure emissivity command
│   ├── transmission.go       # measure transmission command
│   ├── reflectance.go        # measure reflectance command
│   ├── raman.go              # measure raman command (future)
│   └── fluorescence.go       # measure fluorescence/luminescence command (future)
├── cameras/                   # cameras command
│   └── cameras.go
├── freerun/                   # freerun command
│   └── freerun.go
└── internal/
    ├── extract/              # Spectrum extraction from images
    │   ├── extractor.go      # Core extraction interface
    │   ├── window.go         # Spectrum window detection (OBB)
    │   ├── averaging.go      # Row averaging algorithms
    │   └── filtering.go      # Wrapper around filter functions (savgol, gaussian, median, ma)
    ├── calibrate/            # Wavelength calibration orchestration
    │   ├── calibrator.go     # Calibration workflow orchestration
    │   ├── targets.go        # Built-in calibration targets (Hg, Ar, etc.)
    │   ├── matcher.go        # Peak matching with calibration targets
    │   └── quality.go        # R² calculation (uses x/math/mat)
    │   # NOTE: Peak detection uses colorscience.SPD.Peaks()
    │   # NOTE: Polynomial fitting uses colorscience.SPD.Calibrate()
    ├── render/               # Spectrometer-specific visualization
    │   ├── renderer.go       # Spectrum graph rendering
    │   ├── waterfall.go      # Waterfall display
    │   ├── graticule.go      # Grid and labels rendering
    │   ├── overlay.go        # Overlay rendering (illuminants, calibration)
    │   └── colors.go         # Wavelength-to-RGB conversion (uses colorscience)
    ├── correction/           # Spectrum correction and calibration
    │   ├── dark.go           # Dark frame capture and subtraction
    │   ├── reference.go      # Reference/background spectrum handling
    │   ├── mask.go           # Spectrum masking (region-of-interest)
    │   └── normalize.go      # Normalization and linearization
    ├── colorimetry/          # Colorimetry orchestration (uses colorscience)
    │   ├── compute.go        # Wrapper around colorscience functions
    │   ├── display.go        # Colorimetry display overlays
    │   └── illuminants.go    # Wrapper around colorscience.LoadIlluminantSPD()
    │   # NOTE: XYZ uses colorscience.ComputeXYZ()
    │   # NOTE: CCT uses colorscience.ComputeColorTemperature()
    │   # NOTE: CRI uses colorscience.ComputeCRI()
    │   # NOTE: LAB uses xyz.ToLAB()
    ├── measure/              # Measurement modes
    │   ├── single.go         # Single-shot measurement
    │   ├── averaged.go       # Frame averaging
    │   ├── transient.go      # Time-series capture
    │   └── export.go         # Data export via marshallers (CSV, JSON, proto, images, video)
    └── config/               # Configuration management
        ├── config.go         # Config protobuf message handling
        ├── loader.go         # Config loading via marshallers (proto/YAML/JSON)
        └── saver.go          # Config saving via marshallers (proto/YAML/JSON)
```

### Component Integration

The application integrates with EasyRobot packages:

- **`cmd/display/source`**: Reuses camera source abstraction for frame capture (camera sources only, no need for internal camera package)
- **`cmd/display/destination`**: Reuses display destination for window management and video export (no need for internal display package)
- **`x/devices/as734x`**: AS734x spectrum sensor driver (reads spectral bands directly, requires spectral reconstruction)
- **`x/math/mat`**: Matrix operations for polynomial fitting, calibration matrices
- **`x/math/vec`**: Vector operations for spectrum data, wavelength arrays
- **`x/math/colorscience`**: SPD handling, wavelength-to-RGB, colorimetry calculations (CCT, XYZ, LAB, CRI), spectral reconstruction from sensor bands, peak detection (`SPD.Peaks()`), wavelength calibration (`SPD.Calibrate()`)
- **`x/math/filter`**: Digital filters
  - `ma.MovingAverage` - Moving average filter (**already exists** ✅)
  - `savgol.SavitzkyGolay()` - Savitzky-Golay filter (**to be implemented in `filter/savgol`**)
  - `gaussian.Gaussian()` - Gaussian smoothing (**to be implemented in `filter/gaussian`**)
  - `median.Median()` - Median filter (**to be implemented in `filter/median`**)
- **`x/math/dsp`**: DSP algorithms (FFT, convolution, window functions)
- **`x/marshaller/gocv`**: Image encoding/decoding, camera frame marshalling, GUI controls (trackbars, buttons - **to be implemented in `gocv/controls/`**), window management
- **`x/marshaller/proto`**: Protobuf marshalling for configuration and measurements
- **`x/marshaller/yaml`**: YAML marshalling for configuration (human-readable)
- **`x/marshaller/json`**: JSON marshalling for configuration and measurements
- **`x/marshaller/csv`**: CSV marshalling for measurements (vectors, matrices, tensors) - **Needs to be implemented**
- **`x/devices/spectrometer`**: Hardware illuminant control (LED drivers) - **To be implemented for hardware integration**

### Data Flow

**Camera Source Flow** (`--camera` flag):
```
┌─────────────┐
│   Camera    │ (via display/source, --camera flag)
│  - Controls │ (gocv trackbars: exposure, gain, etc.)
└──────┬──────┘
       │ Frame (gocv.Mat)
       ▼
┌─────────────┐
│  Extractor  │ (extract spectrum from pixels)
│  - Window   │ (OBB detection/usage)
│  - Average  │ (row averaging)
│  - Filter   │ (Savitzky-Golay)
│  - Mask     │ (region-of-interest)
└──────┬──────┘
       │ Intensity Vector (vec.Vector)
       ▼
┌─────────────┐
│ Correction  │
│  - Dark     │ (dark measurement subtraction, included in SpectrumMeasurement.dark_spectrum)
│  - Reference│ (reference/calibration spectrum for transmission/reflectance)
│  - Linearize│ (camera linearization, camera only)
└──────┬──────┘
       │ Corrected Intensity Vector
       ▼
┌─────────────┐
│ Calibration │ (if configured, pixel → wavelength mapping)
│  - Wavelength│ (polynomial fitting)
└──────┬──────┘
       │ SPD (colorscience.SPD)
```

**Sensor Source Flow** (`--sensor` flag, e.g., `as734x:/dev/i2c-1:0x39`):
```
┌─────────────┐
│   Sensor    │ (via x/devices/as734x, --sensor flag)
│  - AS734x   │ (reads spectral bands directly, no pixel extraction needed)
│  - Controls │ (integration time, gain)
└──────┬──────┘
       │ Band Measurements (RawMeasurement from as734x)
       ▼
┌─────────────┐
│ Reconstruct │ (via x/math/colorscience)
│  - Band→SPD │ (spectral reconstruction from 8-18 bands to full spectrum)
└──────┬──────┘
       │ SPD (colorscience.SPD) - reconstructed spectrum
       ▼
┌─────────────┐
│ Correction  │
│  - Dark     │ (dark measurement subtraction, included in SpectrumMeasurement.dark_spectrum)
│  - Reference│ (reference/calibration spectrum for transmission/reflectance)
└──────┬──────┘
       │ Corrected SPD
       ▼
┌─────────────┐
│ Calibration │ (wavelength mapping for reconstructed spectrum output)
│  - Wavelength│ (polynomial fitting, still needed for reconstructed spectrum)
└──────┬──────┘
       │ Calibrated SPD (colorscience.SPD)
```

**Common Flow** (after extraction/reconstruction):
```
       │ SPD (colorscience.SPD)
       │ Row 0: wavelengths (vec.Vector)
       │ Row 1: intensities (vec.Vector)
       ▼
┌─────────────┐
│ Colorimetry │ (optional, if enabled)
│  - CCT      │
│  - XYZ/LAB  │
└──────┬──────┘
       │ SPD + Colorimetry Data
       ▼
┌─────────────┐
│  Measure    │
│  - Single   │
│  - Average  │
│  - Transient│
└──────┬──────┘
       │ SpectrumMeasurement
       │ - spectrum: main measurement
       │ - dark_spectrum: dark measurement (if dark correction enabled)
       │ - reference_spectrum: calibration/reference spectrum
       │   * For transmission: illuminant-only spectrum
       │   * For reflectance: white reference standard spectrum
       │   * Documented in metadata if present
       ▼
┌─────────────┐
│  Render     │ (spectrometer-specific)
│  - Graph    │
│  - Waterfall│
│  - Overlay  │ (illuminants, calibration lines)
│  - Colorimetry│ (CCT, XYZ/LAB display)
└──────┬──────┘
       │ Rendered Frame (gocv.Mat)
       ▼
┌─────────────┐
│  Display    │ (via display/destination)
│  - Window   │ (width/height, -1 = fullscreen, show_title, show_controls)
│  - Controls │ (display toggles: CCT, XYZ/LAB, overlays)
└──────┬──────┘
       │
┌──────▼──────┐
│   Export    │
│  - CSV/JSON │
│  - PNG/VID  │
└─────────────┘
```

**Notes**:
- **Camera sources**: Extract spectrum from pixels (window detection, row averaging, filtering)
- **Sensor sources**: Read spectral bands directly, reconstruct full spectrum using `x/math/colorscience.reconstruction`
- **Sensors still need calibration**: Wavelength mapping for reconstructed spectrum output
- **Dark measurement**: Always included in `SpectrumMeasurement.dark_spectrum` if dark correction enabled
- **Reference/Calibration spectrum**: For transmission/reflectance, `SpectrumMeasurement.reference_spectrum` contains the calibration spectrum (illuminant-only for transmission, white reference for reflectance), documented in measurement metadata

---

## Command-Line Flags

### Flag Reuse from display/source and display/destination

The spectrometer application integrates with `cmd/display/source` and `cmd/display/destination` packages, which register the following flags. These flags are reused by spectrometer commands and must not be duplicated:

**Flags from `cmd/display/source`**:
- `--camera <id>` (FlagArray): Camera device ID (can repeat) - **Used for camera source selection**
- `--width <n>` (int): Frame width for cameras (default: 640) - **Used for camera resolution**
- `--height <n>` (int): Frame height for cameras (default: 480) - **Used for camera resolution**
- `--list-cameras` (bool): List all available cameras and exit - **Used by `cameras` command**
- `--images <path>` (FlagArray): Image file paths - **Not used by spectrometer**
- `--video <path>` (FlagArray): Video file paths - **Not used by spectrometer**
- `--generate` (bool): Generate test frames - **Not used by spectrometer**
- `--interest <route>` (FlagArray): DNDM interest routes - **Not used by spectrometer**

**Note**: Camera flags are used when `--camera` is specified. Sensor sources use `--sensor` flag instead (see below).

**Flags from `cmd/display/destination`**:
- `--no-display` (bool): Omit display window - **Used to disable display**
- `--title <string>` (string): Display window title (default: "Display") - **Used for window title**
- `--window-width <n>` (int): Display window width (0 = auto, -1 = fullscreen) - **Used for window size**
- `--window-height <n>` (int): Display window height (0 = auto, -1 = fullscreen) - **Used for window size**
- `--output <path>` (string): Output video file path - **Used for video export (measure/freerun)**
- `--intent <route>` (FlagArray): DNDM intent routes - **Not used by spectrometer**

**Note**: Window width/height of -1 means fullscreen mode. Display settings in config can override these flags.

### Spectrometer-Specific Flags

The following flags are unique to spectrometer commands and do not conflict:

**Source selection flags**:
- `--camera <id>`: Use camera source (via `cmd/display/source`, requires extraction from pixels)
- `--sensor <spec>`: Use spectrum sensor source (e.g., `as734x:/dev/i2c-1:0x39`, requires spectral reconstruction, not pixel extraction)
  - Format: `<type>:<device_path>:<address>` (e.g., `as734x:/dev/i2c-1:0x39`)
  - Supported sensors: `as734x` (AS7341/AS7343, via `x/devices/as734x`)
  - Sensors read spectral bands directly and require reconstruction to full spectrum via `x/math/colorscience`
  - Sensors still need wavelength calibration for reconstructed spectrum output
- `--device <type>`: Use spectrometer device source (e.g., `cr30`, future: `as734x`, etc.)
  - Format: `<type>` (e.g., `cr30`)
  - Device-specific configuration: CR30 uses `--port` and `--baud` flags (similar to `cmd/cr30`)
  - Supported devices: `cr30` (CR30 colorimeter, via `x/devices/cr30`)
  - Devices provide direct spectrum measurements (no extraction or reconstruction needed)
  - Devices may still need wavelength calibration for output

**Calibration flags**:
- `--target <name>`: Calibration target (hg, ar, neon, custom)
- `--skip-window`: Skip window detection if OBB already in config
- `--skip-peaks`: Skip peak detection if calibration points exist
- `--min-peaks <n>`: Minimum peaks required
- `--config-output <file>`: Save config to file instead of stdout (**Note**: Renamed from `--output` to avoid conflict)

**Measurement flags**:
- `--mode <single|average|transient|continuous>`: Measurement mode
- `--frames <n>`: Number of frames for averaging
- `--rate <fps>`: Capture rate for transient mode
- `--duration <seconds>`: Duration for transient mode
- `--spectrum-display <graph|waterfall|dual>`: Display mode (**Note**: Renamed from `--display` to avoid confusion)
- `--export <format>`: Export format (csv, json, png, video, all)
- `--export-path <path>`: Export file path or directory (**Note**: Renamed from `--output` to avoid conflict)
- `--config <file>`: Config file path
- `--patches <path>`: Color patch file path (Proto/JSON/YAML/CSV) - for patch-based measurement and comparison
  - Format auto-detected from file extension (`.pb`, `.json`, `.yaml`, `.csv`) or specified with `--output=<pb,json,yaml,csv>` flag
  - File contains color patches with XYZ, LAB, and optional spectrum values
  - When provided: Display patches, measure against each patch, show comparison (half reference, half measured)
  - Display: RGB color, XYZ/LAB/RGB HEX values (rotate on spacebar), delta E (ΔE)
  - Press "s" key to toggle spectrum overlay on patches (if reference spectrum is available in patch)
  - When not provided: Just measure spectrum, XYZ, LAB, RGB HEX, and show the color
- `--readings <n>`: Number of readings to perform (default: 1)
  - For each reading: Measure spectrum, save spectrum, calculate and save XYZ/LAB values
  - All N spectrums and N XYZ/LAB values are saved

**Device-specific flags** (when `--device=cr30`):
- `--port <path>`: Serial port device (e.g., `/dev/ttyUSB0` or `COM3`)
- `--baud <rate>`: Serial port baud rate (default: 19200)

**Freerun flags**:
- `--config <file>`: Config file path
- `--fullscreen`: Fullscreen display mode
- `--waterfall`: Enable waterfall display

**Note on `--output` flag conflict**: The `--output` flag is registered by `display/destination` for video file output. To avoid conflicts:
- For config export in `calibrate`: Use `--config-output <file>`
- For measurement export: Use `--export-path <path>` (for files) and reuse `--output <path>` from destination for video export
- For video recording in `measure`/`freerun`: Reuse `--output <path>` from destination

**Note on `--display` flag**: To avoid confusion with destination display settings, spectrometer uses `--spectrum-display` for selecting graph/waterfall/dual display mode, while destination's display flags (`--no-display`, `--title`, `--window-width`, `--window-height`) control the window itself.

**Common flags** (all commands):
- `-v=N` or `--verbose=N`: Set log verbosity level (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)
- `-vv`: Shortcut for `-v=4` (TRACE level, maximum verbosity)
- Uses `log/slog` package for structured logging

---

## Commands

### `spectrometer measure` (First Phase - CR30 Device Support)

**Purpose**: Measure spectrum from direct spectrometer devices (starting with CR30).

**This is the FIRST PHASE implementation**, focusing on:
- Framework for spectrometer device "obtainers" (devices that can obtain spectra)
- Spectrum renderer (graph display with colorimetry overlays)
- CR30 device support

**Usage**: `spectrometer measure -device=cr30 [options]`

**Features**:

1. **Basic Measurement** (without `-patches`):
   - Measure spectrum from device
   - Calculate and display XYZ, LAB, RGB HEX values
   - Display RGB color swatch
   - Display spectrum graph with colorimetry overlay

2. **Patch-Based Measurement** (with `-patches=path`):
   - Read color patches from YAML/JSON file (patch XYZ/LAB values)
   - Display patches as color swatches
   - For each patch:
     - Measure spectrum
     - Calculate XYZ, LAB, RGB HEX values
     - Calculate delta E (ΔE) between reference patch and measured values
     - Display patch as half reference (left) and half measured (right)
     - Display RGB color
     - Display XYZ, LAB, RGB HEX values (rotate on spacebar)
     - Display delta E (ΔE)
   - Iterate through all patches

3. **Multiple Readings** (with `--readings=N`):
   - Perform N measurements
   - Save N spectrums (CSV/JSON/Proto)
   - Calculate and save N XYZ/LAB values
   - Each reading saved with timestamp and metadata

**Flags**:
- `--device <type>`: Device type (`cr30` for CR30 colorimeter) - **Required**
- `--patches <path>`: Color patch file path (Proto/JSON/YAML/CSV) - **Optional**
  - Format auto-detected from file extension (`.pb`, `.json`, `.yaml`, `.csv`) or specified with `--output=<pb,json,yaml,csv>` flag
- `--readings <n>`: Number of readings to perform (default: 1)
- `--port <path>`: Serial port device (required for CR30)
- `--baud <rate>`: Serial port baud rate (default: 19200 for CR30)
- `--export <format>`: Export format (csv, json, proto, png)
- `--export-path <path>`: Export file path or directory
- `--output <pb|json|yaml|csv>`: Output format override (for configs, measurements, patches)
  - If not specified, format is auto-detected from file extension (`.pb`, `.json`, `.yaml`, `.csv`)
  - Overrides file extension detection when saving files
- `--display`: Enable display window (default: enabled)
- `--no-display`: Disable display window

**Patch File Format** (Proto/JSON/YAML/CSV):

Patch files can be in any supported format (proto, JSON, YAML, CSV). The format is auto-detected from file extension (`.pb`, `.json`, `.yaml`, `.csv`), or can be explicitly specified with `--output=<pb,json,yaml,csv>`.

**Proto Schema** (`ColorPatches` message):
- `patches`: List of `ColorPatch` messages
- Each `ColorPatch` contains:
  - `name`: Patch identifier
  - `xyz`: Reference XYZ values (3-element vector)
  - `lab`: Reference LAB values (3-element vector)
  - `spectrum`: Optional reference spectrum (row 0: wavelengths, row 1: intensities)

**YAML/JSON Example**:

```yaml
patches:
  - name: "Patch 1"
    xyz: [95.0, 100.0, 108.88]  # Reference XYZ values
    lab: [100.0, 0.0, 0.0]      # Reference LAB values
    spectrum:                    # Optional reference spectrum
      wavelengths: [380, 390, ..., 750]  # Row 0: wavelengths (nm)
      values: [0.1, 0.15, ..., 0.8]      # Row 1: intensities
  - name: "Patch 2"
    xyz: [90.0, 95.0, 100.0]
    lab: [95.0, 5.0, -5.0]
    # spectrum is optional - if not provided, only XYZ/LAB are used
```

**Display Features**:

- **Spectrum Graph**: Wavelength vs. intensity plot
- **Color Swatch**: RGB color representation
- **Colorimetry Overlay**: XYZ, LAB, RGB HEX values
  - Rotate display on spacebar: XYZ → LAB → RGB HEX → repeat
- **Patch Comparison** (with `-patches`):
  - Half reference (left) and half measured (right) patch display
  - Delta E (ΔE) calculation and display
  - CIE76 delta E formula: `ΔE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)`
  - Press "s" key to toggle spectrum overlay on patches (if reference spectrum is available)

**Implementation Notes**:
- Uses `x/devices/cr30` package for CR30 device communication
- Uses `x/math/colorscience` for XYZ/LAB/RGB calculations
- Uses `cmd/display/destination` for display window
- Uses marshallers for patch file loading/saving: `x/marshaller/proto`, `x/marshaller/json`, `x/marshaller/yaml`, `x/marshaller/csv`
- Patch format auto-detected from file extension (`.pb`, `.json`, `.yaml`, `.csv`) or specified via `--output` flag
- Framework should be extensible to support other devices (e.g., `as734x`)

---

### `spectrometer cameras`

**Purpose**: List all available cameras with their capabilities.

**Behavior**:
- Enumerates all video devices (via gocv or v4l2)
- For each camera, displays:
  - Device ID and name
  - Supported resolutions (width × height)
  - Supported frame rates (FPS)
  - Supported pixel formats (MJPEG, YUV, RGB, etc.)
  - Current/default settings if camera is accessible

**Output Format**:
```
Camera 0: /dev/video0
  Name: UVC Camera
  Resolutions:
    - 800×600 @ 30fps, 15fps (MJPEG)
    - 640×480 @ 30fps, 15fps, 10fps (MJPEG)
    - 1280×720 @ 15fps (MJPEG)
  Formats: MJPEG, YUYV

Camera 1: /dev/video2
  Name: Raspberry Pi Camera Module v3
  Resolutions:
    - 1920×1080 @ 30fps, 15fps (SRGGB10_CSI2)
    - 800×600 @ 30fps, 15fps (SRGGB10_CSI2)
  Formats: SRGGB10_CSI2
```

**Implementation Notes**:
- Reuses `source.ListCameras()` from `cmd/display/source` package
- Uses `--list-cameras` flag from `display/source` (automatically handled)
- Calls `source.RegisterAllFlags()` to register camera enumeration flags
- Uses `gocv` unmarshaller for camera enumeration
- On Linux, can use `v4l2-ctl` output parsing as fallback if gocv unavailable

**Flags**:
- **Reused from display/source**: `--list-cameras` (list cameras and exit)

---

### `spectrometer calibrate camera` (Phase 2+)

**Note**: Camera calibration is a Phase 2+ feature. Phase 0 focuses on device support (CR30).

**Purpose**: Interactive calibration workflow to establish wavelength-to-pixel mapping and detect spectrum window.

**Workflow**:

1. **Window Detection** (if OBB not in config):
   - Displays live camera feed
   - User adjusts camera focus/alignment
   - Detects spectrum strip using activity analysis:
     - Computes variance along vertical axis
     - Finds horizontal strip with highest variance (spectrum activity)
     - Estimates orientation (horizontal alignment detection)
     - Creates Oriented Bounding Box (OBB)
   - User confirms or manually adjusts OBB
   - Saves OBB to config

2. **Peak Detection** (if calibration points incomplete):
   - Displays live spectrum with peak detection overlay
   - User provides known light source (Hg lamp default)
   - Algorithm detects peaks:
     - Applies Savitzky-Golay filter
     - Detects local maxima above threshold
     - Enforces minimum distance between peaks
   - Highlights detected peaks on display

3. **Peak Matching** (interactive):
   - User selects calibration target (default: Hg emission lines)
   - Algorithm attempts automatic matching:
     - Sorts peaks by intensity
     - Matches to nearest expected wavelength from target
     - Validates matches (wavelength order, reasonable distances)
   - User confirms matches or manually corrects
   - User can add additional calibration points manually

4. **Polynomial Fitting**:
   - Collects pixel indices and matched wavelengths
   - Fits polynomial:
     - 2nd order if 3 points
     - 3rd order if 4+ points (recommended)
   - Computes R² for quality assessment
   - Displays calibration quality metrics

5. **Validation**:
   - Overlays predicted wavelengths on spectrum
   - User verifies against known lines
   - Allows recalibration if unsatisfactory

6. **Config Export**:
   - Outputs complete config using marshaller
   - Format auto-detected from file extension (`.pb`, `.json`, `.yaml`, `.csv`) or specified with `--output=<pb,json,yaml>` flag
   - If `--config-output` not specified, outputs to stdout (default: YAML)
   - User can redirect to file: `spectrometer calibrate camera > config.yaml`
   - Or specify file with format: `spectrometer calibrate camera --config-output=config.pb`

**Flags**:
- `--target <name>`: Calibration target (hg, ar, neon, custom). Default: hg
- `--skip-window`: Skip window detection if OBB already in config
- `--skip-peaks`: Skip peak detection if calibration points exist
- `--min-peaks <n>`: Minimum peaks required (default: 3)
- `--config-output <file>`: Save config to file instead of stdout (**Note**: Avoids conflict with destination's `--output`)
  - Format auto-detected from file extension (`.pb`, `.json`, `.yaml`)
  - Can override format with `--output=<pb,json,yaml>` flag
- **Reused from display/source**: `--camera <id>`, `--width <n>`, `--height <n>`
- **Reused from display/destination**: `--title <string>`, `--window-width <n>`, `--window-height <n>`, `--no-display`

**Calibration Targets**:

Built-in targets include common emission line sources:

- **Hg (Mercury)**: Default, common in fluorescent lamps
  - 405.4 nm, 436.6 nm, 546.1 nm, 577.0 nm, 579.1 nm, 623.4 nm
- **Ar (Argon)**: 
  - 415.9 nm, 427.2 nm, 451.1 nm, 459.0 nm, 514.5 nm
- **Ne (Neon)**:
  - 540.1 nm, 585.2 nm, 588.2 nm, 594.5 nm, 603.0 nm, 616.4 nm
- **Custom**: User-provided wavelength list (e.g., laser lines)

---

### `spectrometer measure emissivity`

**Purpose**: Measure emissivity spectrum (self-emitting light sources).

**Use Cases**:
- LEDs and laser diodes
- Fluorescent lamps
- Gas discharge tubes
- Any light-emitting source

**Features**:
- Same measurement modes as base `measure` command (single, averaged, transient, continuous)
- No illuminant required (source emits its own light)
- Direct spectrum measurement

**Flags**: Same as base measurement flags (see "Base Measurement Features" below)

---

### `spectrometer measure transmission`

**Purpose**: Measure transmission spectrum through a sample (requires illuminant).

**Use Cases**:
- Optical filters
- Colored glass/plastic
- Liquid samples in cuvettes
- Transparent materials

**Features**:
- **Illuminant Selection** (hardware control, to be implemented):
  - **D65**: Standard daylight (three LEDs)
  - **R, G, B individual**: Red, Green, Blue LEDs individually
  - **RGB single**: All RGB LEDs simultaneously
  - **UV**: Ultraviolet illumination
- Measurement modes: single, averaged, transient, continuous
- Computes transmission as: `transmission = (sample / reference) × 100%`
- Requires reference measurement (illuminant only, no sample)

**Workflow**:
1. Measure reference spectrum (illuminant only)
2. Place sample in beam path
3. Measure sample spectrum
4. Calculate transmission: `T = (I_sample / I_reference)`

**Flags**:
- `--illuminant <d65|r|g|b|rgb|uv>`: Illuminant type (default: d65)
- `--reference <file>`: Path to reference spectrum file (if pre-captured)
- `--capture-reference`: Capture reference spectrum before measurement
- Plus all base measurement flags

---

### `spectrometer measure reflectance`

**Purpose**: Measure reflectance spectrum from a sample surface (requires illuminant).

**Use Cases**:
- Surface colors and materials
- Paint and coatings
- Textiles and fabrics
- Spectral reflectance for colorimetry

**Features**:
- **Illuminant Selection** (hardware control, to be implemented):
  - **D65**: Standard daylight (three LEDs) - **Primary illuminant for reflectance**
  - **R, G, B individual**: Red, Green, Blue LEDs individually (for specialized applications)
  - **RGB single**: All RGB LEDs simultaneously (for specialized applications)
  - **UV**: Ultraviolet illumination (for fluorescence)
- Measurement modes: single, averaged, transient, continuous
- Computes reflectance as: `reflectance = (sample / reference) × 100%`
- Requires reference measurement (white reference standard)

**Workflow**:
1. Measure reference spectrum (white reference standard with illuminant)
2. Measure sample spectrum with illuminant
3. Calculate reflectance: `R = (I_sample / I_reference)`

**Flags**:
- `--illuminant <d65|r|g|b|rgb|uv>`: Illuminant type (default: d65, **recommended for reflectance**)
- `--reference <file>`: Path to white reference spectrum file (if pre-captured)
- `--capture-reference`: Capture white reference spectrum before measurement
- Plus all base measurement flags

---

### `spectrometer measure raman` (Future)

**Purpose**: Raman spectroscopy measurements (requires RGB LED illuminants).

**Use Cases**:
- Molecular identification
- Chemical analysis
- Material characterization

**Features**:
- **Illuminant Selection** (hardware control):
  - **R, G, B individual**: Individual LED excitation (different Raman shifts)
  - **RGB single**: Simultaneous RGB excitation (combined Raman spectrum)
- Different behavior for individual vs. single RGB modes:
  - **Individual**: Measure separate Raman spectra for each LED
  - **RGB single**: Measure combined Raman spectrum
- Measurement modes: single, averaged, transient
- Requires reference measurement (background with illuminant but no sample)

**Workflow**:
1. Measure background spectrum (illuminant only, no sample)
2. Place sample in beam path
3. Measure Raman spectrum with selected illuminant(s)
4. Calculate Raman signal: `Raman = I_sample - I_background`

**Flags**:
- `--illuminant <r|g|b|rgb>`: Illuminant type (**Note**: D65 not used for Raman)
- `--reference <file>`: Path to background spectrum file
- `--capture-reference`: Capture background spectrum before measurement
- Plus all base measurement flags

---

### `spectrometer measure fluorescence` (Future)

**Purpose**: Fluorescence and luminescence measurements with UV excitation and decay analysis.

**Use Cases**:
- Fluorescent materials
- Phosphorescent materials
- Luminescence lifetime analysis
- Stokes shift measurements

**Features**:
- **UV Illuminant**: Ultraviolet LED excitation (hardware control, to be implemented)
- **Special Mode**: Flash-excite and measure decay
  1. Flash UV illuminant (configurable duration)
  2. Measure spectrum immediately after flash
  3. Measure decay over time (time-series)
  4. Analyze decay time constants
- **Decay Analysis**:
  - Capture transient spectrum during decay
  - Fit exponential decay curves
  - Calculate lifetime constants (τ)
- Measurement modes: single flash, averaged flash, decay transient

**Workflow**:
1. **Flash mode**:
   - Flash UV illuminant (default: 100ms)
   - Measure spectrum at t=0 (immediately after flash)
   - Measure transient spectrum during decay
2. **Continuous mode**:
   - Continuous UV illumination
   - Measure steady-state fluorescence spectrum

**Flags**:
- `--illuminant uv`: UV illuminant (required for fluorescence)
- `--mode <flash|continuous>`: Measurement mode (default: continuous)
- `--flash-duration <ms>`: Flash duration in milliseconds (default: 100ms)
- `--decay-duration <seconds>`: Duration to measure decay (default: 5s)
- `--decay-rate <fps>`: Sampling rate for decay measurement (default: 10fps)
- Plus all base measurement flags

---

### Base Measurement Features (Common to All Measure Commands)

All `measure` subcommands support these common features:

1. **Measurement Modes**:
   - **Single**: Capture one spectrum snapshot
   - **Averaged**: Capture N frames and average (reduces noise)
   - **Transient**: Time-series capture at specified rate
   - **Continuous**: Stream measurements until stopped

2. **Display Options**:
   - **Graph**: Standard spectrum plot with graticule
   - **Waterfall**: Time-series visualization (color-coded by wavelength)
   - **Dual**: Both graph and waterfall simultaneously

3. **Averaging**:
   - Configurable frame count (default: 10)
   - Running average display option
   - Outlier rejection (optional)

4. **Export Formats** (via marshallers):
   - **CSV** (`.csv`): Vectors/matrices exported via CSV marshaller (with header support)
   - **Proto** (`.pb`, `.proto`): Binary format for measurements (SpectrumMeasurement protobuf)
   - **JSON** (`.json`): Human-readable format with metadata
   - **YAML** (`.yaml`, `.yml`): Human-readable format with metadata
   - **PNG** (`.png`): Graph image with annotations (via gocv)
   - **Video** (`.mp4`, etc.): Recorded waterfall or animated spectrum (via display/destination)
   - Format is auto-detected from file extension, or can be specified with `--output=<pb,json,yaml,csv>` flag
   - Configs, measurements, and patches all support the same format detection mechanism

5. **Interactive Controls** (via gocv GUI):
   - **Camera Controls** (trackbars):
     - Exposure (seconds) - **Critical for spectroscopy**
     - Gain
     - Brightness/Contrast
     - Saturation (if color camera)
   - **Processing Controls** (trackbars):
     - Savitzky-Golay window size
     - Savitzky-Golay polynomial order
     - Peak detection threshold
     - Peak minimum distance
   - **Display Controls** (toggles/buttons):
     - Show/hide CCT calculation
     - Show/hide XYZ/LAB colorimetry values
     - Show/hide overlays (illuminants, calibration lines)
     - Peak labeling toggle
     - Peak hold mode
     - Dark frame capture button
     - Reference/background capture button
   - **Mouse Interactions**:
     - Cursor wavelength display
     - Click to measure wavelength/intensity

**Common Flags** (all measure subcommands):
- `--mode <single|average|transient|continuous>`: Measurement mode
- `--frames <n>`: Number of frames for averaging (default: 10)
- `--rate <fps>`: Capture rate for transient mode
- `--duration <seconds>`: Duration for transient mode
- `--spectrum-display <graph|waterfall|dual>`: Display mode
- `--export <format>`: Export format (csv, json, proto, png, video, all)
- `--export-path <path>`: Export file path or directory
- `--config <file>`: Config file path (required)
- **Reused from display/source**: `--camera <id>`, `--width <n>`, `--height <n>`
- **Reused from display/destination**: `--output <path>` (for video export), `--title <string>`, `--window-width <n>`, `--window-height <n>`, `--no-display`

**Export Formats** (see full details in Configuration section):
- **CSV**: Via CSV marshaller with `WithHeader` option
- **Proto**: `SpectrumMeasurement` message (includes measurement type, illuminant, dark_spectrum, reference_spectrum)
- **JSON**: Human-readable format with same structure as proto

**Measurement Data**:
- `spectrum`: Main measurement spectrum (wavelengths and intensities)
- `dark_spectrum`: Dark measurement (always included if dark correction enabled)
- `reference_spectrum`: Calibration/reference spectrum (for transmission/reflectance)
  - For transmission: illuminant-only spectrum
  - For reflectance: white reference standard spectrum
  - Documented in metadata if present

---

### `spectrometer freerun` (Phase 5+)

**Note**: Camera-based freerun is a Phase 5+ feature. Phase 0 focuses on device support (CR30).

**Purpose**: Simple real-time spectrum display with minimal overhead.

**Features**:
- Live spectrum capture and display
- Basic interactivity (peak labels, cursor)
- No data logging (streaming only)
- Minimal configuration required

**Use Cases**:
- Quick alignment/verification
- Educational demonstrations
- Real-time monitoring

**Flags**:
- `--config <file>`: Config file path (required)
- `--fullscreen`: Fullscreen display mode
- `--waterfall`: Enable waterfall display
- **Reused from display/source**: `--camera <id>`, `--width <n>`, `--height <n>`
- **Reused from display/destination**: `--output <path>` (for video recording), `--title <string>`, `--window-width <n>`, `--window-height <n>`, `--no-display`

**Interactive Controls**:
- **GUI Controls** (gocv trackbars/buttons):
  - Camera exposure (seconds) - **Critical adjustment**
  - Camera gain
  - Processing filter settings
  - Display toggles (CCT, XYZ/LAB, overlays, peak labels)
  - Capture buttons (dark frame, reference/background)

**Keyboard Controls**:
- `q`: Quit
- `h`: Toggle peak hold
- `m`: Toggle measurement cursor
- `p`: Toggle peak labels
- `s`: Save current frame (graph + CSV)
- `d`: Capture dark frame (with shutter closed/light blocked)
- `r`: Capture reference/background spectrum
- `c`: Toggle CCT display
- `x`: Toggle XYZ/LAB display
- `o`: Toggle overlays (illuminants, calibration lines)

---

## Configuration

Configuration is defined as a protobuf message (`SpectrometerConfiguration`) and can be serialized/deserialized using marshallers (proto, YAML, JSON). Default path: `~/.spectrometer/config.yaml` or `./spectrometer-config.yaml`.

### Protobuf Definition

The configuration is defined in `proto/types/spectrometer/config.proto` (in the project root `proto/types/spectrometer/` directory):

```protobuf
syntax = "proto3";

package types.spectrometer;

option go_package = "github.com/itohio/EasyRobot/types/spectrometer";

import "types/math/vectors.proto";
import "types/math/matrices.proto";
import "google/protobuf/timestamp.proto";

message SpectrometerConfiguration {
  CameraSettings camera = 1;
  Linearization linearization = 2;
  SpectrumWindow window = 3;
  WavelengthCalibration calibration = 4;
  WavelengthRange wavelength = 5;
  DisplaySettings display = 6;
  DarkFrameCorrection dark_frame = 7;
  ReferenceSpectrum reference = 8;
  SignalProcessing processing = 9;
  PeakDetection peaks = 10;
  SpectrumMask mask = 11;
  IlluminantSettings illuminant = 12;  // Hardware illuminant control settings
  repeated ReferenceSpectrumEntry reference_spectrums = 13;  // List of reference spectrums (matrices or paths)
}

message CameraSettings {
  int32 id = 1;
  int32 width = 2;
  int32 height = 3;
  string format = 4;              // MJPEG, YUYV, RGB, etc.
  int32 fps = 5;
  optional float gain = 6;
  optional int64 exposure_us = 7; // Exposure time in microseconds
}

message Linearization {
  repeated double coefficients = 1; // Polynomial coefficients
}

message SpectrumWindow {
  Point2D center = 1;
  Size2D size = 2;
  double rotation_deg = 3;
}

message Point2D {
  int32 x = 1;
  int32 y = 2;
}

message Size2D {
  int32 width = 1;
  int32 height = 2;
}

message WavelengthCalibration {
  types.math.Vector polynomial = 1; // Polynomial coefficients
  repeated CalibrationPoint points = 2;
  double r_squared = 3;
  string method = 4;                // polynomial_2nd_order, polynomial_3rd_order
}

message CalibrationPoint {
  int32 pixel = 1;
  double wavelength = 2;
}

message WavelengthRange {
  double min = 1;
  double max = 2;
}

message DisplaySettings {
  int32 width = 1;
  int32 height = 2;
  GraticuleSettings graticule = 3;
  ColorimetrySettings colorimetry = 4;
  OverlaySettings overlays = 5;
}

message GraticuleSettings {
  int32 major_interval = 1;  // Major grid lines (nm)
  int32 minor_interval = 2;  // Minor grid lines (nm)
}

message ColorimetrySettings {
  bool show_cct = 1;
  bool show_xyz_lab = 2;
}

message OverlaySettings {
  bool enabled = 1;
  repeated string illuminants = 2;  // D65, D50, A, etc.
  bool calibration_lines = 3;
}

message DarkFrameCorrection {
  bool enabled = 1;
  string path = 2;                  // Path to saved dark frame SPD
}

message ReferenceSpectrum {
  bool enabled = 1;
  string path = 2;                  // Path to saved reference SPD
  string mode = 3;                  // subtract, normalize, ratio
}

message ReferenceSpectrumEntry {
  string name = 1;                  // Name/identifier for this reference spectrum
  string description = 2;           // Optional description
  oneof data {
    types.math.Matrix spectrum = 3;  // Inline spectrum matrix (row 0: wavelengths, row 1: values)
    string path = 4;                // Path to file containing spectrum (will be loaded by config loader)
  }
  string format = 5;                // Format if path specified: "csv", "json", "proto", "yaml"
}

message SignalProcessing {
  SavitzkyGolayFilter savgol = 1;
  RowAveraging row_averaging = 2;
}

message SavitzkyGolayFilter {
  bool enabled = 1;
  int32 window_size = 2;            // Must be odd
  int32 polynomial_order = 3;       // Must be < window_size - 1
}

message RowAveraging {
  string method = 1;                // mean, median, weighted_mean
  int32 rows = 2;
  string center_row = 3;            // middle, or specific row index
}

message PeakDetection {
  bool enabled = 1;
  double threshold = 2;             // Relative threshold (0-1)
  int32 min_distance = 3;           // Minimum pixels between peaks
  bool label = 4;                   // Show wavelength labels
}

message SpectrumMask {
  bool enabled = 1;
  repeated WavelengthRegion regions = 2;
}

message WavelengthRegion {
  double min = 1;
  double max = 2;
  bool include = 3;                 // true to include, false to exclude
}

message IlluminantSettings {
  IlluminantType type = 1;          // Current illuminant type
  bool enabled = 2;                 // Hardware illuminant enabled
  int32 intensity_percent = 3;      // Illuminant intensity (0-100%)
  // Future: individual LED control, flash timing, etc.
}

enum IlluminantType {
  ILLUMINANT_NONE = 0;              // No illuminant (emissivity measurements)
  ILLUMINANT_D65 = 1;               // D65 daylight (three LEDs)
  ILLUMINANT_R = 2;                 // Red LED only
  ILLUMINANT_G = 3;                 // Green LED only
  ILLUMINANT_B = 4;                 // Blue LED only
  ILLUMINANT_RGB = 5;               // RGB LEDs simultaneously
  ILLUMINANT_UV = 6;                // Ultraviolet LED
}
```

### Measurement Protobuf Definition

Measurements are defined in `proto/types/spectrometer/measurement.proto` (in the project root `proto/types/spectrometer/` directory):

```protobuf
syntax = "proto3";

package types.spectrometer;

import "types/math/matrices.proto";
import "types/math/vectors.proto";
import "google/protobuf/timestamp.proto";
import "types/spectrometer/config.proto";

message SpectrumMeasurement {
  SpectrometerConfiguration config = 1;
  MeasurementMetadata metadata = 2;
  types.math.Matrix spectrum = 3;        // Row 0: wavelengths, Row 1: intensities
  optional ColorimetryData colorimetry = 4;
  optional MeasurementType measurement_type = 5;  // Emissivity, transmission, reflectance, etc.
  optional IlluminantType illuminant = 6;         // Illuminant used for this measurement
  optional types.math.Matrix reference_spectrum = 7;  // Reference spectrum (for transmission/reflectance)
  optional FluorescenceDecayData fluorescence_decay = 8;  // For fluorescence measurements only
}

message MeasurementMetadata {
  google.protobuf.Timestamp timestamp = 1;
  string source_type = 2;                    // Source type: "camera" or "sensor"
  oneof source_id {
    string camera_id = 3;                    // Camera identifier (e.g., "0", "/dev/video0")
    string sensor_id = 4;                    // Sensor identifier (e.g., "as734x:/dev/i2c-1:0x39")
  }
  string config_file = 5;
  MeasurementMode mode = 4;
  int32 frame_count = 5;                 // For averaged measurements
}

enum MeasurementMode {
  MODE_UNSPECIFIED = 0;
  MODE_SINGLE = 1;
  MODE_AVERAGED = 2;
  MODE_TRANSIENT = 3;
  MODE_CONTINUOUS = 4;
}

enum MeasurementType {
  MEASUREMENT_TYPE_UNSPECIFIED = 0;
  MEASUREMENT_TYPE_EMISSIVITY = 1;       // Self-emitting source
  MEASUREMENT_TYPE_TRANSMISSION = 2;     // Transmission through sample
  MEASUREMENT_TYPE_REFLECTANCE = 3;      // Reflection from sample
  MEASUREMENT_TYPE_RAMAN = 4;            // Raman scattering
  MEASUREMENT_TYPE_FLUORESCENCE = 5;     // Fluorescence/luminescence
}

message ColorimetryData {
  double cct = 1;                        // Correlated Color Temperature
  types.math.Vector xyz = 2;             // XYZ color values
  types.math.Vector lab = 3;             // LAB color values
}

// For fluorescence decay measurements
message FluorescenceDecayData {
  google.protobuf.Timestamp flash_time = 1;
  int64 flash_duration_ms = 2;
  types.math.Matrix initial_spectrum = 3;  // Spectrum immediately after flash
  types.math.Matrix decay_series = 4;      // Time-series of spectra during decay
  repeated double decay_time_constants = 5; // Fitted decay constants (τ values)
}
```

### Configuration Loading/Saving

Configuration is loaded/saved using marshallers with automatic format detection from file extension:

**Format Detection**:
- File extension determines format: `.pb` or `.proto` → proto, `.json` → JSON, `.yaml` or `.yml` → YAML, `.csv` → CSV
- Format can be overridden with `--output=<pb,json,yaml,csv>` flag
- If no extension and no `--output` flag, defaults to YAML for configs

**Supported Formats**:
- **Proto** (`.pb`, `.proto`): Binary format, efficient storage and transmission
- **JSON** (`.json`): Readable, interoperable with other tools
- **YAML** (`.yaml`, `.yml`): Human-readable, easy to edit manually (default for configs)
- **CSV** (`.csv`): For measurements/spectrums only (not for configs)

**Usage**:
```go
// Load configuration - format auto-detected from file extension
config, err := loader.Load(ctx, "config.yaml")    // Loads as YAML
config, err := loader.Load(ctx, "config.json")    // Loads as JSON
config, err := loader.Load(ctx, "config.pb")      // Loads as Proto

// Save configuration - format auto-detected from extension, or use --output flag
err := saver.Save(ctx, "config.yaml", config)     // Saves as YAML
err := saver.Save(ctx, "config.pb", config)       // Saves as Proto
err := saver.Save(ctx, "config", config)          // Saves as YAML (default)
```

**Implementation** (marshaller selection):
```go
// Determine format from extension or flag
format := detectFormat(filename, outputFlag)  // Returns "pb", "json", "yaml", or "csv"

// Select appropriate marshaller
var marshaller Marshaller
switch format {
case "pb", "proto":
    marshaller = proto.NewMarshaller()
case "json":
    marshaller = json.NewMarshaller()
case "yaml", "yml":
    marshaller = yaml.NewMarshaller()
case "csv":
    marshaller = csv.NewMarshaller(csv.WithHeader(true))
}
```

### Configuration Structure (YAML Example)

Configuration can be stored in YAML, JSON, or proto formats. Below is a YAML example (human-readable):

```yaml
# Source Settings (Camera or Sensor)
source:
  type: "camera"                  # Source type: "camera" or "sensor"
  # Camera source (if type == "camera")
  camera:
    id: "0"                       # Camera identifier (e.g., "0", "/dev/video0", "USB:bus/device")
    width: 800                    # Resolution width
    height: 600                   # Resolution height
    format: "MJPEG"               # Pixel format (MJPEG, YUYV, RGB, etc.)
    fps: 30                       # Frame rate
    # Optional camera-specific controls
    gain: 1.0                     # Linear gain
    exposure_us: 33333            # Exposure time (microseconds)
  # Sensor source (if type == "sensor")
  # sensor:
  #   type: "as734x"               # Sensor type (e.g., "as734x")
  #   device_path: "/dev/i2c-1:0x39"  # Device path (I2C bus:address)
  #   integration_time_ms: 100     # Integration time (milliseconds)
  #   gain: 16                     # Sensor gain setting

# Camera Linearization
linearization:
  # Polynomial coefficients for intensity linearization
  # intensity_linear = c0 + c1*intensity + c2*intensity^2 + ...
  coefficients: [0.0, 1.0, 0.0]   # mat.Vector format

# Spectrum Window (Oriented Bounding Box)
window:
  # OBB representation (center, size, rotation)
  center:
    x: 400                        # Pixel coordinates
    y: 300
  size:
    width: 800                    # Spectrum width in pixels
    height: 80                    # Spectrum height (rows to average)
  rotation_deg: 0.0               # Rotation angle (typically 0 for horizontal)
  # Alternative: axis-aligned bounding box
  # bbox: {x: 0, y: 260, width: 800, height: 80}

# Wavelength Calibration
calibration:
  # Polynomial coefficients for pixel -> wavelength mapping
  # wavelength = c0 + c1*pixel + c2*pixel^2 + c3*pixel^3
  polynomial: [380.0, 0.4625, -0.00000125, 0.0]  # mat.Vector
  # Calibration points (pixel index, wavelength)
  points:
    - {pixel: 0, wavelength: 380.0}
    - {pixel: 400, wavelength: 565.0}
    - {pixel: 800, wavelength: 750.0}
  # Quality metrics
  r_squared: 0.999987
  method: "polynomial_3rd_order"

# Wavelength Range
wavelength:
  min: 380.0                      # Minimum wavelength (nm)
  max: 750.0                      # Maximum wavelength (nm)

# Display Settings
display:
  width: 800                      # Window width (pixels), -1 = fullscreen
  height: 320                     # Window height (pixels), -1 = fullscreen
  show_title: true                # Show window title bar
  show_controls: true             # Show GUI controls (trackbars, buttons)
  show_graticule: true            # Show graticule/grid
  graticule:
    major_interval: 50            # Major grid lines (nm)
    minor_interval: 10            # Minor grid lines (nm)
  colorimetry:
    show_cct: false               # Show CCT calculation
    show_xyz_lab: false           # Show XYZ/LAB values
  overlays:
    enabled: false                # Enable overlays
    illuminants: []                # Illuminants to overlay (D65, D50, A, etc.)
    calibration_lines: false      # Show calibration line positions

# Dark Frame Correction
dark_frame:
  enabled: false                  # Enable dark frame subtraction
  path: ""                        # Path to saved dark frame (if not captured)
  # Dark frame is captured as SPD and subtracted from measurements

# Reference/Background Spectrum (single active reference)
reference:
  enabled: false                  # Enable reference spectrum normalization
  path: ""                        # Path to saved reference spectrum
  mode: "subtract"                # subtract, normalize, or ratio
  # Reference spectrum is captured and used for correction in subsequent measurements
  # For transmission: reference = illuminant only
  # For reflectance: reference = white reference standard
  # For Raman: reference = background with illuminant

# Reference Spectrums (library of reference spectrums)
reference_spectrums:
  # List of reference spectrums - can be inline matrices or file paths
  # Config loader will attempt to load paths if provided
  - name: "D65_Standard"
    description: "CIE D65 Standard Illuminant"
    path: "data/illuminants/d65.csv"
    format: "csv"
  - name: "White_Reference"
    description: "White reference standard calibration"
    path: "data/references/white_reference.json"
    format: "json"
  # Inline example (if small enough):
  # - name: "Custom_Reference"
  #   description: "Custom reference spectrum"
  #   spectrum:  # Inline matrix (row 0: wavelengths, row 1: values)
  #     rows: 2
  #     cols: 401
  #     data: [[380.0, 380.5, ...], [1.0, 1.0, ...]]

# Illuminant Hardware Control
illuminant:
  type: "d65"                     # none, d65, r, g, b, rgb, uv
  enabled: false                  # Hardware illuminant enabled
  intensity_percent: 100          # Illuminant intensity (0-100%)
  # Hardware control will be implemented later
  # For now, illuminant selection is documented but not hardware-controlled

# Signal Processing
processing:
  # Savitzky-Golay filter
  savgol:
    enabled: true
    window_size: 17               # Must be odd
    polynomial_order: 7           # Must be < window_size - 1
  # Row averaging
  row_averaging:
    method: "mean"                # mean, median, weighted_mean
    rows: 3                       # Number of rows to average
    center_row: "middle"          # middle, or specific row index

# Peak Detection
peaks:
  enabled: true
  threshold: 0.3                  # Relative threshold (0-1)
  min_distance: 50                # Minimum pixels between peaks
  label: true                     # Show wavelength labels

# Spectrum Masking
mask:
  enabled: false                  # Enable spectrum masking (region-of-interest)
  regions: []                     # List of wavelength ranges to include/exclude
  # Example: [{min: 400, max: 500, include: true}, {min: 700, max: 750, include: false}]
```

### Configuration Validation

- Camera ID must be valid (check via `cameras` command)
- Resolution must be supported by camera
- Window must be within image bounds
- Calibration polynomial order matches number of points
- Wavelength range is valid (typically 350-1000 nm)

---

## Implementation Details

### Code Reuse and Separation of Concerns

**IMPORTANT**: The spectrometer application reuses existing implementations from shared packages and does NOT reimplement algorithms that already exist. See `REVIEW.md` for detailed analysis.

**Key Reuses**:
- **Peak Detection**: Uses `colorscience.SPD.Peaks(threshold, minProminence)` from `x/math/colorscience`
- **Wavelength Calibration**: Uses `colorscience.SPD.Calibrate(pairs ...float32)` from `x/math/colorscience`
- **Filtering**: Uses filters from `x/math/filter`:
  - `savgol.SavitzkyGolay()` - Savitzky-Golay filter (to be implemented)
  - `gaussian.Gaussian()` - Gaussian smoothing (to be implemented)
  - `median.Median()` - Median filter (to be implemented)
  - `ma.MovingAverage` - Moving average (**already exists** ✅)
- **CRI Calculations**: Uses `colorscience.ColorScience.ComputeCRI(spd)` from `x/math/colorscience`
- **XYZ/LAB Calculations**: Uses `colorscience.ComputeXYZ()`, `xyz.ToLAB()` from `x/math/colorscience`
- **GUI Controls**: Uses `gocv/controls.Trackbar` and `gocv/controls.Button` from `x/marshaller/gocv/controls` (to be implemented)

**Spectrometer-Specific**:
- Window detection from image variance (camera-specific)
- Row averaging algorithms (spectrum extraction-specific)
- Calibration workflow orchestration (user interaction, target matching)
- Spectrum visualization (graph, waterfall, overlays)
- Measurement orchestration (single, averaged, transient workflows)

### Spectrum Extraction

**Window Detection** (when OBB not provided):

1. Capture reference frame (averaged over N frames)
2. Convert to grayscale
3. Compute vertical variance:
   ```go
   variance[i] = var(frame[column i, :])
   ```
4. Find horizontal strip with highest average variance
5. Detect orientation (using Hough transform or line fitting)
6. Create OBB from detected strip

**Row Averaging**:

- Extract N rows around center row (default: 3 rows, ±1)
- Average intensities per column: `intensity[i] = mean(window[i, center±offset])`
- Options:
  - Mean: Simple average
  - Median: Robust to outliers
  - Weighted mean: Center row weighted higher

**Filtering**:

- **Savitzky-Golay**: Uses `savgol.SavitzkyGolay(signal, windowSize, polynomialOrder)` from `x/math/filter/savgol` (to be implemented)
  - Smoothing while preserving peak shape
  - Parameters: window_size (odd), polynomial_order (< window_size - 1)
- **Gaussian**: Uses `gaussian.Gaussian(signal, sigma, windowSize)` from `x/math/filter/gaussian` (to be implemented)
- **Median**: Uses `median.Median(signal, windowSize)` from `x/math/filter/median` (to be implemented)
- **Moving Average**: Uses `ma.MovingAverage` from `x/math/filter/ma` (**already exists** ✅)
- Applied after averaging, before display

**Note**: Filtering is a general signal processing operation, not spectrometer-specific. Filters belong in `x/math/filter` package, not reimplemented in spectrometer.

**Intensity Linearization**:

If camera nonlinearity coefficients provided:
```go
intensity_linear = c0 + c1*intensity + c2*intensity² + ...
```

### Calibration

**Wavelength Calibration**:

Uses `colorscience.SPD.Calibrate(pairs ...float32)` from `x/math/colorscience`:

```go
// Calibrate SPD using known (index, wavelength) pairs
calibrated, err := spd.Calibrate(
    0, 400.0,  // Index 0 = 400nm
    400, 565.0, // Index 400 = 565nm
    800, 750.0, // Index 800 = 750nm
)
// Uses cubic Catmull-Rom spline interpolation between calibration points
```

**Peak Detection**:

Uses `colorscience.SPD.Peaks(threshold, minProminence)` from `x/math/colorscience`:

```go
// Detect peaks in spectrum
peaks := spd.Peaks(0.5, 0.1) // Peaks with value >= 0.5 and prominence >= 0.1
for _, peak := range peaks {
    // peak.Index, peak.Wavelength, peak.Value, peak.Prominence
}
```

**Auto-Detection**:

Uses `colorscience.SPD.DetectCalibrationPoints(referenceSPD, minConfidence)` for automatic calibration point detection:

```go
// Match measured SPD to reference SPD with known wavelengths
referenceSPD, _ := colorscience.LoadIlluminantSPD("D65")
calPoints := measuredSPD.DetectCalibrationPoints(referenceSPD, 0.8)
// Returns []CalibrationPoint with (index, wavelength) pairs
```

**Spectrometer-Specific**:

- Calibration targets database (Hg, Ar, Ne emission line wavelengths)
- Peak matching algorithm (match detected peaks to known calibration targets)
- Quality assessment (R² calculation using `x/math/mat` for polynomial fitting)

**Note**: Peak detection and wavelength calibration are general SPD operations, implemented in `x/math/colorscience`. The spectrometer orchestrates these operations but does not reimplement them.

**Peak Matching**:

1. Sort detected peaks by intensity
2. Sort calibration target wavelengths
3. Attempt greedy matching:
   - Match strongest peak to nearest target wavelength
   - Continue with remaining peaks
   - Validate wavelength order (monotonic)
4. User confirmation/editing interface

### Colorimetry

**Colorimetry Calculations**:

Uses `x/math/colorscience` for all colorimetry calculations:

- **XYZ**: `colorscience.ColorScience.ComputeXYZ(spd, wavelengths)` → `XYZ`
- **CCT**: `colorscience.ColorScience.ComputeColorTemperature(spd)` → `(CCT, Duv, error)`
- **CRI**: `colorscience.ColorScience.ComputeCRI(spd)` → `(Ra, error)` ✅
- **LAB**: `xyz.ToLAB(whitePoint)` → `LAB`
- **RGB**: `xyz.ToRGB(out255)` → `RGB`

**Standard Illuminants**:

Uses `colorscience.LoadIlluminantSPD(name)` for standard illuminants:
- D65, D50, A, F11, E, etc.

**Note**: All colorimetry calculations are general color science operations, implemented in `x/math/colorscience`. The spectrometer orchestrates these calculations but does not reimplement them.

### Display

**Graph Rendering**:

- Background: White
- Graticule: Gray grid lines, labeled at major intervals
- Spectrum: Colored vertical lines (wavelength-to-RGB)
- Peak labels: Yellow boxes with wavelength text
- Cursor: Crosshair with wavelength display

**Waterfall**:

- Time on Y-axis (bottom = newest)
- Wavelength on X-axis
- Color intensity represents spectral power
- Color hue represents wavelength (wavelength-to-RGB)
- Configurable history length (default: 320 frames)

**Wavelength-to-RGB**:

Implementation based on CIE color matching:
- Uses `x/math/colorscience` for wavelength → XYZ → RGB conversion
- Gamma correction applied
- Handles UV/IR (gray for invisible wavelengths)

### Integration with display/source and display/destination

**Source Integration**:

```go
// Reuse camera source from display package
cameraSrc := source.NewCameraSource(cameraID, width, height, format, fps)
cameraSrc.Start(ctx)

for {
    frame := cameraSrc.ReadFrame()
    // Extract spectrum from frame
    spectrum := extractor.Extract(frame)
    // ...
}
```

**Destination Integration**:

```go
// Reuse display destination
displayDst := destination.NewDisplayDestination()
displayDst.Start(ctx)

// Create annotated frame
annotatedFrame := renderer.Render(spectrum, displayOptions)
displayDst.AddFrame(annotatedFrame)

// Export to video
if videoOutput {
    videoDst := destination.NewVideoDestination(outputPath, fps)
    videoDst.AddFrame(annotatedFrame)
}
```

---

## Additional Features and Recommendations

### Suggested Enhancements

1. **Automatic Calibration Validation**:
   - Compare detected peaks to expected peaks from calibration target
   - Flag suspicious matches (outlier detection)
   - Suggest recalibration if drift detected

2. **Multi-Camera Support**:
   - Calibrate multiple cameras simultaneously
   - Cross-validation between cameras
   - Average measurements from multiple cameras

3. **Dark Frame Correction**:
   - Capture dark frame (shuttered)
   - Subtract from spectrum: `spectrum_corrected = spectrum - dark`
   - Config option: `dark_frame_path`

4. **Reference Spectrum Normalization**:
   - Normalize to known reference (e.g., D65 illuminant)
   - Store reference spectrum in config
   - Apply: `spectrum_normalized = spectrum / reference`

5. **Integration Time Optimization**:
   - Auto-adjust exposure/gain to prevent saturation
   - Target peak intensity range (e.g., 70-90% of max)

6. **Spectral Resolution Metrics**:
   - Compute FWHM of known narrow lines
   - Report spectral resolution (nm/pixel)
   - Monitor resolution over time (alignment drift)

7. **Batch Processing**:
   - Process directory of images/videos
   - Batch calibration workflow
   - Script generation for automation

8. **Export to Standard Formats**:
   - **CIE SPD**: Standard format for colorimetry
   - **ASCII SPD**: Common spectrometer format
   - **HDF5**: Scientific data format

9. **Colorimetry Integration**:
   - Compute XYZ, LAB, RGB from spectrum
   - Use `x/math/colorscience` for conversions
   - Display color swatch alongside spectrum

10. **Web Interface**:
    - Optional HTTP server for remote control
    - Real-time spectrum streaming (WebSocket)
    - REST API for measurements

11. **Calibration Database**:
    - Store multiple calibration profiles
    - Switch between calibrations
    - Calibration history tracking

12. **Measurement Statistics**:
    - SNR calculation
    - Measurement uncertainty estimation
    - Outlier detection and reporting

### Performance Considerations

- **Frame Processing**: Use goroutines for parallel frame processing
- **Display Updates**: Throttle display updates (e.g., 30 FPS max)
- **Memory**: Limit waterfall history to prevent memory growth
- **Camera Buffers**: Configure appropriate buffer sizes to prevent drops

### Error Handling

- **Camera Errors**: Graceful fallback, retry logic
- **Calibration Errors**: Validate polynomial fit, warn on low R²
- **Window Detection Errors**: Fallback to manual specification
- **Export Errors**: Continue operation, log errors

---

## Required Implementations

### CSV Marshaller/Unmarshaller

The CSV marshaller (`x/marshaller/csv`) needs to be implemented to support spectrometer measurements:

**Requirements**:
- Support for `vec.Vector` types (single column or multi-column vectors)
- Support for `mat.Matrix` types (rows × cols)
- Support for `tensor.Tensor` types (limited to 1D/2D tensors for CSV compatibility)
- Options:
  - `WithHeader(true/false)`: Include column headers as first row
  - `WithZeroRowHeader(true/false)`: For matrices, treat row 0 as header (otherwise row 0 is data)

**Usage Examples**:
```go
// CSV marshaller with header
csvMarshaller := csv.NewMarshaller(csv.WithHeader(true))
err := csvMarshaller.Marshal(writer, spectrumMatrix)  // Matrix with headers

// CSV unmarshaller with header
csvUnmarshaller := csv.NewUnmarshaller(csv.WithHeader(true))
var spectrumMatrix mat.Matrix
err := csvUnmarshaller.Unmarshal(reader, &spectrumMatrix)
```

**CSV Format**:
- Vector export: Single column or named columns (with header)
- Matrix export: Rows × columns, optionally with row 0 as header
- Header format: Column names in first row (if `WithHeader(true)`)
- For matrices: If `WithZeroRowHeader(true)`, row 0 is header; if false, row 0 is data

### Protobuf Definitions

Protobuf definitions for spectrometer need to be created in the project root `proto/types/spectrometer/` directory:

1. **`proto/types/spectrometer/config.proto`**: `SpectrometerConfiguration` message
2. **`proto/types/spectrometer/measurement.proto`**: `SpectrumMeasurement` message

**Build Process**:
- Proto files are built using `make proto` from the project root
- Generated Go code is placed in `types/spectrometer/` (via go_package option)
- Import path: `github.com/itohio/EasyRobot/types/spectrometer`

See the Configuration section above for the complete protobuf schema definitions.

---

## Open Questions

### Configuration and Calibration

1. **Config File Location**:
   - Should we support multiple config files (profile system)?
   - Default location: `~/.spectrometer/` or current directory?

2. **Calibration Persistence**:
   - Save calibration points separately from full config?
   - Version calibration data format for migration?

3. **Camera Settings Persistence**:
   - Save optimal gain/exposure per camera?
   - Auto-restore settings on startup?

### User Interface

4. **Interactive Calibration**:
   - CLI-only or require GUI library?
   - Should we use terminal UI (tui) or require display?

5. **Window Management**:
   - Multiple windows (graph + waterfall) or single window?
   - How to handle window resizing with fixed aspect ratio?

6. **Keyboard Controls**:
   - Standardize across all commands?
   - Configurable key bindings?

### Algorithm Choices

7. **Peak Detection Sensitivity**:
   - Auto-tune threshold based on signal characteristics?
   - User-adjustable during calibration?

8. **Polynomial Order Selection**:
   - Auto-select order based on point count or always allow override?
   - Support higher-order polynomials (4th, 5th) for wide ranges?

9. **Window Detection Robustness**:
   - What if spectrum is not perfectly horizontal?
   - Support rotated/non-rectangular windows?

10. **Linearization**:
    - Required for all cameras or optional?
    - How to measure/derive linearization coefficients?

### Integration

11. **DNDM Integration**:
    - Should measurements be publishable via DNDM intent?
    - Real-time spectrum streaming via DNDM?

12. **External Calibration Sources**:
    - Import calibration from external file?
    - Share calibrations between instruments?

### Data Export

13. **Timestamp Precision**:
    - Include timestamps in exports?
    - Precision: seconds, milliseconds, nanoseconds?

14. **Metadata Extensibility**:
    - Allow custom metadata in JSON exports?
    - Tags/annotations for measurements?

15. **Video Codec**:
    - Preferred codec for video exports (H.264, VP9, raw)?
    - Frame rate for video export?

### Scientific Accuracy

16. **Uncertainty Propagation**:
    - Calculate and report measurement uncertainty?
    - Include uncertainty in export formats?

17. **Calibration Traceability**:
    - Link calibration to reference standards?
    - Document calibration procedure in metadata?

18. **Spectrum Interpolation**:
    - Resample to standard wavelengths (e.g., 1nm intervals)?
    - Preserve original pixel resolution?

### Testing and Validation

19. **Test Data**:
    - Provide sample calibration data?
    - Synthetic spectrum generator for testing?

20. **Validation Metrics**:
    - Define acceptable R² for calibration?
    - Warnings for suspicious measurements?

---

## Reference Implementation Details

### PySpectrometer2 Algorithms

**Spectrum Extraction** (lines 176-234):
- Fixed crop window: `y = frameHeight/2 - 40`, `h = 80`
- 3-row averaging: `(row[halfway-1] + row[halfway] + row[halfway+1]) / 3`
- Peak hold: max intensity per pixel

**Filtering** (lines 262-265):
- Savitzky-Golay: window_size=17, poly_order=7
- Applied when peak hold is OFF

**Calibration** (specFunctions.py, lines 241-370):
- 2nd order polynomial for 3 points
- 3rd order polynomial for 4+ points
- R² calculation for quality assessment

**Peak Detection** (specFunctions.py, lines 147-238):
- First-order difference method
- Minimum distance enforcement
- Relative threshold

**Display** (lines 197-341):
- 320px tall graph
- Graticule: 10nm minor, 50nm major
- Wavelength-to-RGB coloring
- Peak labels with flagpoles

### Go Implementation Notes

- Use `gocv` for camera capture and image processing
- Use `x/math/mat` for polynomial fitting (Vandermonde + least squares)
- Use `x/math/vec` for spectrum data (1D arrays)
- Use `x/math/colorscience.SPD` for calibrated spectra (2-row matrix)
- Follow `cmd/display` patterns for source/destination integration

---

## Success Criteria

1. **Functional**:
   - ✅ All four commands implemented and working
   - ✅ Accurate wavelength calibration (R² > 0.999 for 4+ points)
   - ✅ Real-time display at ≥30 FPS (camera-limited)
   - ✅ Reliable spectrum extraction across different setups

2. **Usability**:
   - ✅ Clear CLI interface with helpful error messages
   - ✅ Comprehensive configuration file with defaults
   - ✅ Interactive calibration workflow
   - ✅ Professional export formats

3. **Quality**:
   - ✅ Modular, testable code following SOLID principles
   - ✅ Integration with EasyRobot packages
   - ✅ Documentation and examples
   - ✅ Error handling and validation

4. **Scientific**:
   - ✅ Measurement accuracy comparable to PySpectrometer2
   - ✅ Reproducible results via configuration
   - ✅ Proper uncertainty handling

---

## Timeline and Milestones

1. **Phase 1: Core Infrastructure** (Week 1-2)
   - Project structure and configuration system
   - Camera integration (display/source)
   - Basic spectrum extraction

2. **Phase 2: Calibration** (Week 3-4)
   - Window detection
   - Peak detection and matching
   - Polynomial fitting
   - Calibration command

3. **Phase 3: Display and Measurement** (Week 5-6)
   - Graph rendering
   - Waterfall display
   - Measure command
   - Freerun command

4. **Phase 4: Polish** (Week 7-8)
   - Export formats
   - Error handling
   - Documentation
   - Testing

---

## Conclusion

This specification defines a comprehensive, professional-grade spectrometer application built on EasyRobot's scientific computing foundation. The modular architecture ensures maintainability and extensibility, while the command-based interface provides flexibility for various use cases.

The specification is intended to be a living document, evolving as implementation progresses and new requirements emerge. Open questions should be resolved during design discussions and early implementation phases.

