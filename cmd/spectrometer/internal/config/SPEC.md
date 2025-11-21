# Config Module - Specification

## Overview

The config module manages spectrometer configuration loading, saving, validation, and reference spectrum path resolution. Configuration is defined as a protobuf message (`SpectrometerConfiguration`) and can be serialized/deserialized using marshallers (proto, YAML, JSON).

## Responsibilities

1. **Configuration Structure**: Work with `types.spectrometer.SpectrometerConfiguration` protobuf message
2. **Loading**: Load configuration from proto/YAML/JSON files via marshallers
3. **Saving**: Save configuration to proto/YAML/JSON files via marshallers
4. **Validation**: Validate configuration parameters (ranges, relationships, file existence)
5. **Reference Spectrum Resolution**: Resolve paths in `reference_spectrums` list and load spectra
6. **Default Configuration**: Provide sensible defaults for unset parameters

## Interfaces

```go
// Loader loads configuration from various formats
type Loader interface {
    Load(ctx context.Context, path string) (*types.spectrometer.SpectrometerConfiguration, error)
    LoadFromReader(ctx context.Context, r io.Reader, format string) (*types.spectrometer.SpectrometerConfiguration, error)
}

// Saver saves configuration to various formats
type Saver interface {
    Save(ctx context.Context, path string, config *types.spectrometer.SpectrometerConfiguration) error
    SaveToWriter(ctx context.Context, w io.Writer, format string, config *types.spectrometer.SpectrometerConfiguration) error
}

// Validator validates configuration parameters
type Validator interface {
    Validate(ctx context.Context, config *types.spectrometer.SpectrometerConfiguration) error
}

// ReferenceResolver resolves and loads reference spectrum paths
type ReferenceResolver interface {
    Resolve(ctx context.Context, entry *types.spectrometer.ReferenceSpectrumEntry) (mat.Matrix, error)
    ResolveAll(ctx context.Context, entries []*types.spectrometer.ReferenceSpectrumEntry) (map[string]mat.Matrix, error)
}
```

## Reference Spectrum Loading

The config module must handle reference spectrum entries that can be:

1. **Inline Matrix**: Already loaded in `spectrum` field
2. **File Path**: Path to file, must be loaded from disk

**Loading Strategy**:
- Try to load path if provided
- Support formats: CSV, JSON, proto, YAML
- Use appropriate marshallers based on `format` field
- Cache loaded spectra in memory
- Return error if path cannot be resolved or loaded

**File Format Detection**:
- Primary: Auto-detect from file extension (`.pb` or `.proto` → proto, `.json` → JSON, `.yaml` or `.yml` → YAML, `.csv` → CSV)
- Override: Use `format` parameter if explicitly provided (from `--output` flag)
- Default: YAML for configs if cannot be determined

## Configuration Validation

Validate the following:

- Camera ID must be valid (check via camera enumeration)
- Resolution must be supported by camera
- Window must be within image bounds
- Calibration polynomial order matches number of points
- Wavelength range is valid (typically 350-1000 nm)
- Reference spectrum paths exist (if specified)
- Savitzky-Golay window_size is odd and > polynomial_order
- Row averaging parameters are valid
- Peak detection parameters are in valid ranges

## Default Configuration

Provide defaults for:

- Camera: id="0", width=800, height=600, format="MJPEG", fps=30
- Window: center at image center, size from config or auto-detect
- Display: width=800, height=320
- Processing: Savitzky-Golay enabled, window_size=17, polynomial_order=7
- Row averaging: method="mean", rows=3, center_row="middle"
- Peak detection: enabled=true, threshold=0.3, min_distance=50

## Implementation Details

### Loading

```go
// Load configuration from file (auto-detect format from extension)
config, err := loader.Load(ctx, "config.yaml")   // Detects YAML from .yaml
config, err := loader.Load(ctx, "config.json")   // Detects JSON from .json
config, err := loader.Load(ctx, "config.pb")     // Detects Proto from .pb

// Load configuration from reader with explicit format (override extension)
config, err := loader.LoadFromReader(ctx, reader, "yaml")
```

### Saving

```go
// Save configuration to file (format auto-detected from extension)
err := saver.Save(ctx, "config.yaml", config)     // Saves as YAML
err := saver.Save(ctx, "config.json", config)     // Saves as JSON
err := saver.Save(ctx, "config.pb", config)       // Saves as Proto

// Save configuration to writer with explicit format (override extension)
err := saver.SaveToWriter(ctx, writer, "yaml", config)
```

**Note**: Format can be overridden via `--output=<pb,json,yaml>` command-line flag, which takes precedence over file extension detection.

### Reference Spectrum Resolution

```go
// Resolve single reference spectrum entry
spectrum, err := resolver.Resolve(ctx, entry)

// Resolve all reference spectrums in config
spectra, err := resolver.ResolveAll(ctx, config.ReferenceSpectrums)
```

## Error Handling

- **File Not Found**: Return clear error with suggestion to create default config
- **Invalid Format**: Return error indicating expected format
- **Validation Errors**: Return detailed validation errors listing all issues
- **Reference Path Resolution**: Return error with path that failed, continue with others if possible

## Dependencies

- `types/spectrometer` - Protobuf generated types
- `x/marshaller/proto` - Proto marshalling
- `x/marshaller/yaml` - YAML marshalling
- `x/marshaller/json` - JSON marshalling
- `x/marshaller/csv` - CSV marshalling (for reference spectrum loading)
- `x/math/mat` - Matrix types
- `log/slog` - Structured logging

## Testing

- Unit tests for loading/saving each format
- Unit tests for validation logic
- Unit tests for reference spectrum resolution
- Integration tests with actual config files
- Error handling tests

