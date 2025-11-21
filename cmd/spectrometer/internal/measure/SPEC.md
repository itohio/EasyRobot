# Measure Module - Specification

## Overview

The measure module handles different measurement modes (single, averaged, transient, continuous) and data export via marshallers (CSV, JSON, proto, images, video). It coordinates spectrum extraction, correction, colorimetry, and export.

## Responsibilities

1. **Single Measurement**: Capture single spectrum from frame
2. **Averaged Measurement**: Average multiple frames for noise reduction
3. **Transient Measurement**: Capture time-series spectra
4. **Continuous Measurement**: Stream measurements continuously
5. **Export**: Export measurements via marshallers (CSV, JSON, proto, PNG, video)

## Interfaces

```go
// Measurement captures a spectrum measurement
type Measurement struct {
    SPD             colorscience.SPD
    Timestamp       time.Time
    Metadata        *types.spectrometer.MeasurementMetadata
    Colorimetry     *types.spectrometer.ColorimetryData
}

// SingleMeasurer captures single-shot measurements
type SingleMeasurer interface {
    Measure(ctx context.Context, frame gocv.Mat) (*Measurement, error)
}

// AveragedMeasurer captures averaged measurements
type AveragedMeasurer interface {
    Measure(ctx context.Context, frames []gocv.Mat) (*Measurement, error)
}

// TransientMeasurer captures time-series measurements
type TransientMeasurer interface {
    Measure(ctx context.Context, duration time.Duration, rate float64, frameChan <-chan gocv.Mat) ([]*Measurement, error)
}

// Exporter exports measurements via marshallers
type Exporter interface {
    Export(ctx context.Context, measurement *Measurement, format string, path string) error
    ExportMultiple(ctx context.Context, measurements []*Measurement, format string, path string) error
}
```

## Single Measurement

**Workflow**:
1. Extract spectrum from frame
2. Apply corrections (dark frame, reference, masking)
3. Apply calibration (pixel → wavelength)
4. Calculate colorimetry (if enabled)
5. Return measurement

## Averaged Measurement

**Workflow**:
1. Capture N frames
2. Extract spectrum from each frame
3. Apply corrections to each spectrum
4. Average spectra (wavelength-by-wavelength)
5. Apply calibration
6. Calculate colorimetry
7. Return measurement

**Averaging Method**:
- Simple mean (default)
- Median (robust to outliers)
- Weighted mean (if needed)

## Transient Measurement

**Workflow**:
1. Start time-series capture
2. Capture frames at specified rate
3. Extract and process each spectrum
4. Store measurements with timestamps
5. Stop after specified duration
6. Return measurement series

**Use Cases**:
- Fluorescence decay analysis
- Time-varying sources
- Dynamic measurements

## Continuous Measurement

**Workflow**:
1. Stream frames continuously
2. Extract and process spectra in real-time
3. Display in real-time (no export)
4. Stop on user command

**Use Cases**:
- Real-time monitoring
- Live display
- Interactive analysis

## Export

**Format Detection**:
- Format is auto-detected from file extension: `.pb` or `.proto` → proto, `.json` → JSON, `.yaml` or `.yml` → YAML, `.csv` → CSV
- Format can be overridden with `--output=<pb,json,yaml,csv>` flag (from command line)
- If no extension and no format flag, defaults to JSON for measurements

**Formats**:
- **CSV** (`.csv`): Wavelength, intensity columns (via CSV marshaller)
- **JSON** (`.json`): Full measurement data (via JSON marshaller)
- **YAML** (`.yaml`, `.yml`): Full measurement data (via YAML marshaller)
- **Proto** (`.pb`, `.proto`): Binary format (via proto marshaller)
- **PNG** (`.png`): Spectrum graph image (via gocv)
- **Video** (`.mp4`, etc.): Animated spectrum display (via display/destination)

**Export Implementation**:
```go
// Export single measurement - format auto-detected from extension
err := exporter.Export(ctx, measurement, "", "measurement.csv")   // Detects CSV from .csv
err := exporter.Export(ctx, measurement, "", "measurement.json")  // Detects JSON from .json
err := exporter.Export(ctx, measurement, "", "measurement.pb")    // Detects Proto from .pb

// Export with explicit format override (if --output flag provided)
err := exporter.Export(ctx, measurement, "json", "measurement")   // Uses JSON format

// Export multiple measurements (transient)
err := exporter.ExportMultiple(ctx, measurements, "", "measurements.json")
```

**CSV Format**:
```csv
wavelength,intensity
380.0,0.0
380.5,0.1
...
```

**JSON Format**:
- Full `SpectrumMeasurement` protobuf message (as JSON)
- Includes metadata, colorimetry, etc.

**Proto Format**:
- Binary `SpectrumMeasurement` protobuf message

**PNG Format**:
- Rendered spectrum graph (via render module)

**Video Format**:
- Recorded spectrum display (via display/destination video writer)

## Dependencies

- `x/math/colorscience` - SPD handling
- `x/marshaller/csv` - CSV export
- `x/marshaller/json` - JSON export
- `x/marshaller/proto` - Proto export
- `gocv` - Image/video export
- `cmd/display/destination` - Video export
- `internal/extract` - Spectrum extraction
- `internal/correction` - Spectrum correction
- `internal/calibrate` - Wavelength calibration
- `internal/colorimetry` - Colorimetry calculations
- `internal/render` - Graph rendering (for PNG export)
- `types/spectrometer` - Measurement types
- `log/slog` - Structured logging

## Testing

- Unit tests for single measurement
- Unit tests for averaged measurement
- Unit tests for transient measurement
- Unit tests for export (each format)
- Integration tests with real frames
- Test data generation

