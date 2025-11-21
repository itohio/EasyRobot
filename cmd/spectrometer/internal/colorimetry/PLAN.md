# Colorimetry Module - Implementation Plan

## Tasks

### 1. Standard Illuminants
- [ ] **USE `colorscience.LoadIlluminantSPD(name)`** (already implemented)
- [ ] Create wrapper function for spectrometer overlay rendering
- [ ] Support custom illuminants (user-provided SPDs via config)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 2. XYZ Calculation
- [ ] **USE `colorscience.ColorScience.ComputeXYZ(spd, wavelengths)`** (already implemented)
- [ ] Create wrapper function for spectrometer configuration
- [ ] Support different observers (via ColorScience configuration)
- [ ] Handle edge cases (spectrometer-specific error handling)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 3. CCT Calculation
- [ ] **USE `colorscience.ColorScience.ComputeColorTemperature(spd)`** (already implemented)
- [ ] Create wrapper function for spectrometer configuration
- [ ] Handle edge cases (spectrometer-specific error handling)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 4. LAB Calculation
- [ ] **USE `xyz.ToLAB(whitePoint)`** (already implemented)
- [ ] Create wrapper function for spectrometer white point configuration
- [ ] Support different white points (via ColorScience configuration)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 4b. CRI Calculation ✅
- [ ] **USE `colorscience.ColorScience.ComputeCRI(spd)`** (already implemented) ✅
- [ ] Create wrapper function for spectrometer display
- [ ] Handle edge cases (invalid CCT, insufficient samples)
- [ ] **DO NOT reimplement** - use existing implementation from `x/math/colorscience`

### 5. RGB Calculation (Optional)
- [ ] Implement RGB calculation from XYZ
- [ ] Support sRGB color space
- [ ] Support other color spaces (Adobe RGB, etc.)
- [ ] Apply gamma correction

### 6. Colorimetry Calculator Integration
- [ ] Implement ColorimetryCalculator interface
- [ ] Combine all calculations
- [ ] Error handling and logging

### 7. Display Overlays
- [ ] Implement overlay renderer
- [ ] Render CCT, XYZ, LAB text
- [ ] Render RGB swatch (optional)
- [ ] Position overlays correctly
- [ ] Semi-transparent background

### 8. Testing
- [ ] Unit tests for XYZ calculation (known SPDs)
- [ ] Unit tests for CCT calculation (known SPDs)
- [ ] Unit tests for LAB calculation (known SPDs)
- [ ] Unit tests for illuminant loading
- [ ] Integration tests with real spectra

## Implementation Order

1. Standard illuminants (simple data loading)
2. XYZ calculation (foundation, uses colorscience)
3. LAB calculation (depends on XYZ)
4. CCT calculation (depends on XYZ)
5. RGB calculation (optional, depends on XYZ)
6. Colorimetry calculator integration
7. Display overlays
8. Testing

## Research Needed

- `x/math/colorscience` API for colorimetry calculations
- CCT calculation algorithms
- Standard illuminant data formats
- RGB color space conversion matrices

## Dependencies

- `x/math/colorscience` must be available
- Render module for overlay rendering
- Config module for colorimetry settings

