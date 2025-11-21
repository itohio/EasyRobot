# Correction Module - Specification

## Overview

The correction module applies spectrum corrections: dark frame subtraction, reference spectrum normalization, spectrum masking, and intensity linearization. These corrections improve measurement accuracy and account for systematic errors.

## Responsibilities

1. **Dark Frame Correction**: Subtract dark frame (shutter closed) from spectra
2. **Reference Spectrum Handling**: Apply reference spectrum normalization/subtraction
3. **Spectrum Masking**: Apply region-of-interest filtering (wavelength ranges)
4. **Normalization**: Normalize spectra to reference or illuminant
5. **Linearization**: Apply camera response linearization

## Interfaces

```go
// DarkFrameCorrector applies dark frame subtraction
type DarkFrameCorrector interface {
    Capture(ctx context.Context, frames []gocv.Mat) (colorscience.SPD, error)
    Apply(ctx context.Context, spd colorscience.SPD, dark colorscience.SPD) (colorscience.SPD, error)
    Load(ctx context.Context, path string) (colorscience.SPD, error)
    Save(ctx context.Context, path string, dark colorscience.SPD) error
}

// ReferenceCorrector applies reference spectrum correction
type ReferenceCorrector interface {
    Capture(ctx context.Context, frames []gocv.Mat) (colorscience.SPD, error)
    Apply(ctx context.Context, spd colorscience.SPD, reference colorscience.SPD, mode string) (colorscience.SPD, error)
    Load(ctx context.Context, path string) (colorscience.SPD, error)
    Save(ctx context.Context, path string, reference colorscience.SPD) error
}

// MaskApplier applies spectrum masking
type MaskApplier interface {
    Apply(ctx context.Context, spd colorscience.SPD, mask *types.spectrometer.SpectrumMask) (colorscience.SPD, error)
}

// Normalizer normalizes spectra
type Normalizer interface {
    Normalize(ctx context.Context, spd colorscience.SPD, reference colorscience.SPD) (colorscience.SPD, error)
    NormalizeToIlluminant(ctx context.Context, spd colorscience.SPD, illuminant string) (colorscience.SPD, error)
}

// Linearizer applies camera response linearization
type Linearizer interface {
    Apply(ctx context.Context, intensity vec.Vector, coefficients vec.Vector) (vec.Vector, error)
}
```

## Dark Frame Correction

**Capture**:
- Capture N frames with shutter closed / light blocked
- Average frames to reduce noise
- Save as SPD (wavelengths and intensities)

**Application**:
- Subtract dark frame: `corrected = spectrum - dark`
- Handle negative values (clamp to 0 or handle properly)

**Workflow**:
1. User presses 'd' or clicks "Capture Dark Frame" button
2. System captures N frames
3. Extract spectrum from frames
4. Average spectra
5. Store in config or save to file

## Reference Spectrum Correction

**Modes**:
- **subtract**: `corrected = spectrum - reference` (for transmission/reflectance)
- **normalize**: `corrected = spectrum / reference` (normalize to reference)
- **ratio**: `corrected = spectrum / reference` (same as normalize, different semantics)

**Application**:
- For transmission: reference = illuminant only
- For reflectance: reference = white reference standard
- For Raman: reference = background with illuminant

**Workflow**:
1. User presses 'r' or clicks "Capture Reference" button
2. System captures reference spectrum
3. Store in config or save to file
4. Apply to subsequent measurements

## Spectrum Masking

**Purpose**: Filter spectrum to region-of-interest (ROI)

**Implementation**:
- Define wavelength regions to include/exclude
- Set values outside ROI to 0 or NaN
- Support multiple regions

**Example**:
- Include: 400-500 nm
- Exclude: 700-750 nm
- Result: Spectrum only in 400-500 nm, masked elsewhere

## Normalization

**Normalize to Reference**:
- Divide spectrum by reference spectrum: `normalized = spectrum / reference`
- Handle division by zero (warn or skip)

**Normalize to Illuminant**:
- Load standard illuminant SPD (D65, D50, A, etc.)
- Divide spectrum by illuminant SPD
- Used for relative spectral power distribution

## Linearization

**Purpose**: Correct camera nonlinear response

**Implementation**:
```go
intensity_linear = c0 + c1*intensity + c2*intensity² + c3*intensity³ + ...
```

- Apply polynomial transformation
- Use coefficients from config

## Dependencies

- `x/math/colorscience` - SPD handling
- `x/math/vec` - Vector operations
- `x/math/mat` - Matrix operations
- `types/spectrometer` - Configuration types
- `x/marshaller/*` - Loading/saving spectra
- `log/slog` - Structured logging

## Testing

- Unit tests for dark frame subtraction
- Unit tests for reference correction modes
- Unit tests for masking
- Unit tests for normalization
- Unit tests for linearization
- Integration tests with real spectra

