# Colorimetry Module - Specification

## Overview

The colorimetry module orchestrates colorimetry calculations by using `x/math/colorscience` functions. It does NOT reimplement color science algorithms - it wraps and coordinates existing implementations from `x/math/colorscience`. The module also provides spectrometer-specific display overlays for colorimetry results.

## Responsibilities

1. **Colorimetry Orchestration**: Coordinate multiple calculations from `x/math/colorscience`
2. **Display Overlays**: Generate spectrometer-specific display overlays for colorimetry results
3. **Wrapper Functions**: Provide spectrometer-specific interfaces wrapping `colorscience` functions

**Note**: All actual calculations (XYZ, CCT, CRI, LAB, RGB) use `x/math/colorscience` implementations directly.

## Interfaces

```go
// ColorimetryCalculator calculates colorimetric values from SPD
type ColorimetryCalculator interface {
    Calculate(ctx context.Context, spd colorscience.SPD) (*ColorimetryResults, error)
    CalculateCCT(ctx context.Context, spd colorscience.SPD) (float64, error)
    CalculateXYZ(ctx context.Context, spd colorscience.SPD) (vec.Vector, error)
    CalculateLAB(ctx context.Context, spd colorscience.SPD) (vec.Vector, error)
}

// ColorimetryResults contains all calculated colorimetry values
type ColorimetryResults struct {
    CCT  float64       // Correlated Color Temperature (Kelvin)
    Duv  float64       // Distance from Planckian locus
    XYZ  vec.Vector    // XYZ tristimulus values (3 elements)
    LAB  vec.Vector    // LAB color values (3 elements)
    RGB  vec.Vector    // RGB color values (optional, 3 elements)
}

// IlluminantProvider provides standard illuminant SPDs
type IlluminantProvider interface {
    Get(ctx context.Context, name string) (colorscience.SPD, error)
    List() []string
}

// DisplayOverlayRenderer renders colorimetry overlays
type DisplayOverlayRenderer interface {
    Render(ctx context.Context, img gocv.Mat, results *ColorimetryResults, config *types.spectrometer.ColorimetrySettings) error
}
```

## CCT Calculation

**Uses `colorscience.ColorScience.ComputeColorTemperature(spd)`**:

```go
// Calculate CCT and Duv from SPD
cct, duv, err := cs.ComputeColorTemperature(spd)
// Returns CCT in Kelvin and Duv (distance from Planckian locus)
```

**Implementation**: The `colorscience.ColorScience.ComputeColorTemperature()` method already implements:
- XYZ calculation from SPD
- Chromaticity coordinate conversion (x, y)
- Planckian locus matching
- CCT interpolation
- Duv calculation

**Spectrometer-Specific**: The colorimetry module wraps this function for spectrometer configuration and error handling.

**Note**: CCT calculation is a general color science operation, implemented in `x/math/colorscience`. Use it, don't reimplement.

## XYZ Calculation

**Uses `colorscience.ColorScience.ComputeXYZ(spd, wavelengths)`**:

```go
// Calculate XYZ from SPD
xyz, err := cs.ComputeXYZ(spd, wavelengths)
// Returns XYZ tristimulus values
// X = xyz[0], Y = xyz[1], Z = xyz[2]
```

**Implementation**: The `colorscience.ColorScience.ComputeXYZ()` method already implements:
- SPD integration with CIE color matching functions
- Observer selection (CIE 1931 2° or CIE 1964 10°)
- Illuminant normalization
- X, Y, Z tristimulus value calculation

**Spectrometer-Specific**: The colorimetry module wraps this function for spectrometer configuration.

**Note**: XYZ calculation is a general color science operation, implemented in `x/math/colorscience`. Use it, don't reimplement.

## LAB Calculation

**Uses `xyz.ToLAB(whitePoint)`**:

```go
// Convert XYZ to LAB
lab := xyz.ToLAB(WhitePointD65_10)
// Returns LAB color values
// L = lab.L(), a = lab.A(), b = lab.B()
```

**Implementation**: The `xyz.ToLAB()` method already implements:
- Standard CIE LAB conversion formulas
- White point normalization
- Nonlinear transformations
- L*, a*, b* value calculation

**Spectrometer-Specific**: The colorimetry module wraps this function for spectrometer white point configuration.

**Note**: LAB calculation is a general color science operation, implemented in `x/math/colorscience`. Use it, don't reimplement.

## RGB Calculation

**Algorithm** (optional):
1. Convert XYZ to RGB using conversion matrix
2. Apply gamma correction
3. Clamp to valid RGB range (0-255 or 0-1)

**Implementation**:
- Use `x/math/colorscience` RGB conversion functions
- Support different RGB color spaces (sRGB, Adobe RGB, etc.)

## CRI Calculation ✅

**Uses `colorscience.ColorScience.ComputeCRI(spd)`**:

```go
// Calculate Color Rendering Index (CRI Ra, 0-100)
cri, err := cs.ComputeCRI(spd)
// Returns general CRI (Ra, average of 8 test color samples)
// Higher values (up to 100) indicate better color rendering
```

**Implementation**: The `colorscience.ColorScience.ComputeCRI()` method already implements:
- CCT calculation for reference illuminant selection
- 8 CIE test color sample comparisons
- Color difference (ΔE) calculations
- Special CRI (Ri) and general CRI (Ra) calculation

**Spectrometer-Specific**: The colorimetry module wraps this function for spectrometer display.

**Note**: CRI calculation is a general color science operation, implemented in `x/math/colorscience`. Use it, don't reimplement.

## Standard Illuminants

**Uses `colorscience.LoadIlluminantSPD(name)`**:

```go
// Load standard illuminant
d65, err := colorscience.LoadIlluminantSPD("D65")
// Returns SPD with wavelengths and values
```

**Available Illuminants**:
- **D65**: Daylight illuminant (6500K) ✅
- **D50**: Daylight illuminant (5000K) ✅
- **A**: Incandescent lamp (2856K) ✅
- **F11**: Fluorescent lamp (4000K)
- **E**: Equal-energy illuminant

**Implementation**: The `colorscience.LoadIlluminantSPD()` function already implements:
- Loading from embedded CSV data
- Return as SPD (wavelengths and values)
- Auto-interpolation to target wavelengths if needed

**Spectrometer-Specific**: The colorimetry module wraps this function for spectrometer overlay rendering.

**Note**: Standard illuminants are general color science data, implemented in `x/math/colorscience`. Use it, don't reimplement.

## Display Overlays

**Components**:
- CCT display: "CCT: 6500K" text
- XYZ display: "XYZ: (95.0, 100.0, 108.9)" text
- LAB display: "LAB: (100.0, 0.0, 0.0)" text
- RGB display: Color swatch + RGB values (optional)

**Positioning**:
- Configurable position (top-left, top-right, etc.)
- Semi-transparent background for readability

## Dependencies

- `x/math/colorscience` - SPD handling, colorimetry calculations
- `x/math/vec` - Vector operations
- `types/spectrometer` - Configuration types
- `gocv` - Display overlay rendering
- `log/slog` - Structured logging

## Testing

- Unit tests for CCT calculation (known SPDs)
- Unit tests for XYZ calculation (known SPDs)
- Unit tests for LAB calculation (known SPDs)
- Unit tests for illuminant loading
- Integration tests with real spectra

