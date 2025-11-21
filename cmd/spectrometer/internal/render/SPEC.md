# Render Module - Specification

## Overview

The render module provides spectrum visualization capabilities, including spectrum graphs, colorimetry overlays, patch comparisons, and color swatches. This module is used by all measurement commands to display spectrum data.

## Responsibilities

1. **Spectrum Graph Rendering**: Render wavelength vs. intensity plots
2. **Colorimetry Overlay Rendering**: Display XYZ, LAB, RGB HEX values
3. **Color Swatch Rendering**: Display RGB color representation
4. **Patch Comparison Rendering**: Display half reference, half measured patches
5. **Delta E Calculation**: Calculate and display color difference (ΔE)

## Interfaces

```go
// Renderer renders spectrum visualizations
type Renderer interface {
    // RenderSpectrum renders a spectrum graph
    RenderSpectrum(ctx context.Context, img gocv.Mat, spd colorscience.SPD, config *types.spectrometer.DisplaySettings) error
    
    // RenderColorimetry renders colorimetry overlay (XYZ, LAB, RGB HEX)
    RenderColorimetry(ctx context.Context, img gocv.Mat, xyz colorscience.XYZ, lab colorscience.LAB, rgb colorscience.RGB, displayMode ColorimetryDisplayMode) error
    
    // RenderColorSwatch renders an RGB color swatch
    RenderColorSwatch(ctx context.Context, img gocv.Mat, rgb colorscience.RGB, x, y, width, height int) error
    
    // RenderPatchComparison renders a patch comparison (half reference, half measured)
    RenderPatchComparison(ctx context.Context, img gocv.Mat, refPatch, measuredPatch Patch, x, y, width, height int) error
}

// ColorimetryDisplayMode specifies what colorimetry values to display
type ColorimetryDisplayMode int

const (
    ColorimetryDisplayXYZ ColorimetryDisplayMode = iota
    ColorimetryDisplayLAB
    ColorimetryDisplayRGBHEX
)

// Patch represents a color patch for comparison
type Patch struct {
    Name       string
    Reference  colorscience.LAB  // Reference LAB values
    Measured   colorscience.LAB  // Measured LAB values
    ReferenceXYZ colorscience.XYZ  // Reference XYZ values
    MeasuredXYZ  colorscience.XYZ  // Measured XYZ values
    DeltaE     float32           // Delta E (ΔE) value
    RGB        colorscience.RGB  // RGB color
}
```

## Spectrum Graph Rendering

**Features**:
- Wavelength vs. intensity plot
- Grid and axis labels
- Color-coded spectrum (wavelength to RGB conversion)
- Configurable scaling (auto-scale or fixed range)

**Implementation**:
- Uses `x/math/colorscience` for wavelength-to-RGB conversion
- Uses gocv for drawing operations
- Supports configurable display settings (width, height, fullscreen)

## Colorimetry Overlay Rendering

**Features**:
- Display XYZ, LAB, RGB HEX values
- Rotate display on spacebar: XYZ → LAB → RGB HEX → repeat
- Semi-transparent overlay background
- Configurable position and font size

**Implementation**:
- Formats values with appropriate precision
- Handles keyboard input for rotation (via display/destination event handling)
- Uses gocv text rendering

## Color Swatch Rendering

**Features**:
- Display RGB color as rectangular swatch
- Border for visibility
- Configurable size and position

**Implementation**:
- Uses gocv rectangle drawing
- Fills rectangle with RGB color (converted from XYZ/LAB)

## Patch Comparison Rendering

**Features**:
- Display patch as half reference (left) and half measured (right)
- Show patch name
- Show delta E (ΔE) value
- Show colorimetry values (XYZ, LAB, RGB HEX) with rotation
- Visual comparison of reference vs. measured colors

**Implementation**:
- Splits patch rectangle into two halves
- Left half: Reference RGB color
- Right half: Measured RGB color
- Overlays text with delta E and colorimetry values
- Handles keyboard input for colorimetry value rotation

## Delta E Calculation

**Formula**: CIE76 delta E
```
ΔE = sqrt((L1 - L2)² + (a1 - a2)² + (b1 - b2)²)
```

**Implementation**:
- Uses `colorscience.LAB` values
- Calculates Euclidean distance in LAB color space
- Displays with appropriate precision (typically 2 decimal places)

## Usage Example

```go
renderer := render.NewRenderer()

// Render spectrum graph
spd := colorscience.NewSPD(wavelengths, intensities)
err := renderer.RenderSpectrum(ctx, img, spd, displaySettings)

// Calculate colorimetry
xyz, _ := cs.ComputeXYZ(spd.Matrix)
lab := xyz.ToLAB(whitePoint)
rgb := xyz.ToRGB(true)

// Render colorimetry overlay
renderer.RenderColorimetry(ctx, img, xyz, lab, rgb, ColorimetryDisplayLAB)

// Render color swatch
renderer.RenderColorSwatch(ctx, img, rgb, 100, 100, 50, 50)

// Render patch comparison
patch := Patch{
    Name: "Patch 1",
    Reference: refLAB,
    Measured: measuredLAB,
    ReferenceXYZ: refXYZ,
    MeasuredXYZ: measuredXYZ,
    DeltaE: deltaE,
    RGB: rgb,
}
renderer.RenderPatchComparison(ctx, img, patch, refPatch, measuredPatch, 200, 200, 100, 50)
```

## Dependencies

- `x/math/colorscience` - Colorimetry calculations, wavelength-to-RGB conversion ✅
- `gocv` - Image drawing operations ✅
- `cmd/display/destination` - Display window and event handling ✅
