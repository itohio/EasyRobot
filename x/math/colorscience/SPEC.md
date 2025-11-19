# math/colorscience – Color Science Specification

## Overview

The `colorscience` package provides comprehensive spectral color calculations for Go, implementing CIE color science standards. It supports conversion between spectral power distributions (SPD) and color spaces (XYZ, LAB, RGB), chromatic adaptation, and advanced features like multi-sensor SPD reconstruction.

The package uses embedded standard observer and illuminant spectral data, following SOLID principles with clear separation between data loading, calculations, and conversions.

## Design Principles

1. **SOLID Compliance** – Standalone functions for pure transformations (XYZ↔LAB, XYZ↔RGB, adaptation); `ColorScience` struct only for stateful spectrum-to-XYZ operations.
2. **Options Pattern** – Configuration via functional options for flexibility and defaults.
3. **Explicit Dependencies** – CMF, illuminant SPD, and white point must be provided explicitly.
4. **Matrix/Vector Operations** – Uses `vec.Vector` and `mat.Matrix` from `x/math` for all calculations.
5. **Embedded Data** – Standard observers and illuminants embedded at compile time via `embed` package.
6. **Zero-Copy When Possible** – SPD interpolation and integration use efficient vector operations.
7. **Type Safety** – `SPD` and `ObserverCMF` are types based on `matTypes.Matrix` interface, allowing methods to be attached while being compatible with any matrix implementation.
8. **Value Semantics** – SPD and CMF are stored and passed as values (not pointers) for simpler ownership and copying.

## Type Catalogue

### Core Types

#### `ColorScience`

The main struct for spectral-to-XYZ calculations. Holds configured CMF, illuminant, white point, and calibration SPDs.

```go
type ColorScience struct {
    cmf            ObserverCMF  // Stored as value, not pointer
    illuminant     SPD          // Stored as value, not pointer
    illuminantName string
    whitePoint     WhitePoint
    whitePointSet  bool         // Track if white point was explicitly set
    observer       ObserverType
    dark           SPD          // Dark calibration SPD (sensor dark reading)
    light          SPD          // Light calibration SPD (reference white/light reading)
}
```

**Semantics:**
- Created via `New(opts...)` with options pattern
- `ComputeXYZ()` is the primary method, uses configured CMF and illuminant
- `Calibrate()` normalizes measurement SPDs using dark and light calibration readings
- CMF and illuminant are automatically loaded based on observer type and illuminant name
- CMF and illuminant are interpolated to match input wavelengths automatically
- White point is used for LAB conversions (via standalone functions)
- Dark and light SPDs are used for radiometric calibration of measurements

#### `SPD` (Spectral Power Distribution)

Represents a spectral distribution as a 2-row matrix. `SPD` embeds `matTypes.Matrix` interface, allowing methods to be attached while implementing the interface (due to embedding).

```go
type SPD struct {
    matTypes.Matrix
}
```

**Matrix Layout:**
- Row 0: wavelengths
- Row 1: values

**Methods:**
- `NewSPD(wavelengths, values)` – Create from vectors (returns SPD)
- `(spd SPD) Wavelengths()` – Get wavelength vector (row 0)
- `(spd SPD) Values()` – Get values vector (row 1)
- `(spd SPD) Len()` – Get number of wavelength/value pairs
- `(spd SPD) Interpolate(targetWavelengths)` – Resample to new wavelength grid using linear interpolation (returns SPD)
- `(spd SPD) Calibrate(pairs ...float32)` – Calibrate wavelengths using known (index, wavelength) pairs with cubic Catmull-Rom spline interpolation (returns SPD)
- `(spd SPD) Reconstruct(sensorResponses, useDampedLS, lambda)` – Reconstruct SPD values from multi-sensor measurements (modifies SPD in place)

**Usage:**
Since `SPD` embeds `matTypes.Matrix`, all matrix interface methods are promoted and available:
```go
spd := NewSPD(wavelengths, values)
wavelengths := spd.Wavelengths()  // Method
rows := spd.Rows()  // Promoted interface method
```

#### `ObserverCMF` (Color Matching Functions)

Contains the x̄, ȳ, z̄ color matching functions as a 4-row matrix. `ObserverCMF` embeds `matTypes.Matrix` interface, allowing methods to be attached while implementing the interface (due to embedding).

```go
type ObserverCMF struct {
    matTypes.Matrix
}
```

**Matrix Layout:**
- Row 0: wavelengths
- Row 1: XBar
- Row 2: YBar
- Row 3: ZBar

**Methods:**
- `LoadCMF(observer)` – Load CMF for specified observer (returns ObserverCMF)
- `(cmf ObserverCMF) Wavelengths()` – Get wavelengths as SPD (row 0 duplicated)
- `(cmf ObserverCMF) WavelengthsValues()` – Get wavelength vector (row 0)
- `(cmf ObserverCMF) XBar()` – Get XBar as SPD
- `(cmf ObserverCMF) XBarValues()` – Get XBar vector (row 1)
- `(cmf ObserverCMF) YBar()` – Get YBar as SPD
- `(cmf ObserverCMF) YBarValues()` – Get YBar vector (row 2)
- `(cmf ObserverCMF) ZBar()` – Get ZBar as SPD
- `(cmf ObserverCMF) ZBarValues()` – Get ZBar vector (row 3)
- `(cmf ObserverCMF) Len()` – Get number of wavelength/CMF pairs

#### `WhitePoint`

CIE XYZ white point values. Type alias for `vec.Vector3D`.

```go
type WhitePoint vec.Vector3D
```

Standard constants provided: `WhitePointD65_10`, `WhitePointD50_10`, `WhitePointA`, etc.

#### `XYZ`

CIE XYZ tristimulus values. Type alias for `vec.Vector3D`.

```go
type XYZ vec.Vector3D
```

**Methods:**
- `NewXYZ(X, Y, Z)` – Create XYZ from components
- `(xyz XYZ) Luminance()` – Get luminance (Y component: `xyz[1]`)
- `(xyz XYZ) ToLAB(illuminant)` – Convert to LAB
- `(xyz XYZ) ToRGB(out255)` – Convert to RGB
- `(xyz XYZ) Adapt(Ws, Wd, method)` – Chromatic adaptation

**Component Access:**
- Access components directly: `xyz[0]` (X), `xyz[1]` (Y), `xyz[2]` (Z)
- Or use Vector3D method: `X, Y, Z := vec.Vector3D(xyz).XYZ()`

#### `LAB`

CIE LAB color values. Type alias for `vec.Vector3D`.

```go
type LAB vec.Vector3D
```

**Methods:**
- `NewLAB(L, a, b)` – Create LAB from components
- `(lab LAB) L()` – Get L component (lightness)
- `(lab LAB) A()` – Get a component (green-red axis)
- `(lab LAB) B()` – Get b component (blue-yellow axis)
- `(lab LAB) LAB()` – Get L, a, b components
- `(lab LAB) ToXYZ(illuminant)` – Convert to XYZ
- `(lab LAB) ToRGB(illuminant, out255)` – Convert to RGB

#### `RGB`

sRGB color values. Type alias for `vec.Vector3D`.

```go
type RGB vec.Vector3D
```

**Methods:**
- `NewRGB(r, g, b)` – Create RGB from components
- `(rgb RGB) R()` – Get R component
- `(rgb RGB) G()` – Get G component
- `(rgb RGB) B()` – Get B component
- `(rgb RGB) RGB()` – Get r, g, b components
- `(rgb RGB) ToXYZ()` – Convert to XYZ
- `(rgb RGB) ToLAB(illuminant)` – Convert to LAB

#### `ObserverType`

Enum for standard observers:
- `Observer2Deg` – CIE 1931 2 Degree Standard Observer
- `Observer10Deg` – CIE 1964 10 Degree Standard Observer

#### `CalibrationPoint`

Represents a calibration point: index → wavelength mapping.

```go
type CalibrationPoint struct {
    Index      int     // Index in the SPD (0-based)
    Wavelength float32 // Actual wavelength in nanometers
}
```

#### `SensorResponse`

Represents a photodetector with known spectral response and measured reading.

```go
type SensorResponse struct {
    Name     string         // Sensor name/identifier
    Response matTypes.Matrix // Spectral response of the sensor (must be 2-row SPD matrix)
    Reading  float32        // Measured response value to the stimulus
}
```

## API Reference

### Construction

#### `New(opts...)`

Creates a `ColorScience` instance with automatic CMF and illuminant loading.

**Parameters:**
- `opts ...Option` – Optional configuration

**Options:**
- `WithObserver(ObserverType)` – Set observer type (default: `Observer10Deg`)
- `WithIlluminant(string)` – Set illuminant by name (default: "D65"), auto-sets white point
- `WithIlluminantSPD(matTypes.Matrix)` – Override illuminant with custom SPD matrix
- `WithWhitePoint(WhitePoint)` – Override white point (default: `WhitePointD65_10`)
- `WithDark(matTypes.Matrix)` – Set dark calibration SPD (sensor dark reading)
- `WithLight(matTypes.Matrix)` – Set light calibration SPD (reference white/light reading)

**Behavior:**
- CMF is automatically loaded for the specified observer type
- Illuminant is automatically loaded by name if not provided via `WithIlluminantSPD`
- All matrices (SPD, CMF) are stored as values (not pointers) and cloned when needed

**Returns:** `(*ColorScience, error)`

**Example:**
```go
cs, err := New(WithObserver(Observer10Deg), WithIlluminant("D65"), WithWhitePoint(WhitePointD65_10))
// Or with defaults:
cs, err := New()  // Uses Observer10Deg, "D65", WhitePointD65_10
```

### Data Loading

#### `LoadCMF(observer)`

Loads Color Matching Functions for the specified observer.

**Parameters:**
- `observer ObserverType` – Observer type (`Observer2Deg` or `Observer10Deg`)

**Returns:** `(ObserverCMF, error)` – Returns full CMF data (value, not pointer)

**Example:**
```go
cmf, err := LoadCMF(Observer10Deg) // Full data
wavelengths := cmf.Wavelengths()  // Access via method
```

#### `LoadIlluminantSPD(illuminantName)`

Loads illuminant SPD by name.

**Parameters:**
- `illuminantName string` – Illuminant name ("D65", "D50", "A")

**Returns:** `(SPD, error)` – Returns full SPD data (value, not pointer)

**Available Illuminants:**
- `"A"` – Standard Illuminant A (incandescent)
- `"D50"` – Daylight 5000K
- `"D65"` – Daylight 6500K (standard for sRGB)

#### `AvailableIlluminants()`

Returns list of available illuminant names.

**Returns:** `[]string`

### Spectrum to XYZ Conversion

#### `ComputeXYZ(spd, wavelengths)`

Computes CIE XYZ tristimulus values from a spectral power distribution.

**Parameters:**
- `spd vec.Vector` – Spectral power distribution values
- `wavelengths vec.Vector` – Wavelength vector (must match spd length)

**Returns:** `(XYZ, error)`

**Algorithm:**
1. Interpolate CMF and illuminant to match input wavelengths (if needed)
2. Build product SPD: `I(λ) = S(λ) * E(λ)` for reflective, `I(λ) = S(λ)` for emissive
3. Compute integrals using integration kernel: `X = k * ∫[I(λ) * x̄(λ)]·dλ`
4. Normalize: `k = 100 / ∫[E(λ) * ȳ(λ)]·dλ` for reflective, `k = 683` for emissive

**Example:**
```go
wavelengths := vec.NewFrom(400, 410, 420, 430, 440, 450)
spd := vec.NewFrom(0.5, 0.6, 0.7, 0.65, 0.55, 0.5)
xyz, err := cs.ComputeXYZ(spd, wavelengths)
lab := xyz.ToLAB(WhitePointD65_10)
rgb := xyz.ToRGB(true) // 0-255 range

// Access components directly via array indexing (XYZ is vec.Vector3D)
X, Y, Z := vec.Vector3D(xyz).XYZ()
// Or access directly: xyz[0], xyz[1], xyz[2]
```

### Luminosity Calculations

#### `Luminance(X, Y, Z)`

Calculates luminance (cd/m²) from CIE XYZ tristimulus values.
Luminance is given by the Y component of XYZ.
For absolute luminance, Y should be in cd/m² (requires calibration).
For relative luminance, Y is normalized (0-100).

**Parameters:**
- `X, Y, Z float32` – CIE XYZ tristimulus values

**Returns:** `float32` – Luminance (Y component in cd/m² or normalized 0-100)

#### `(cs *ColorScience) LuminanceFromSPD(spd, wavelengths)`

Calculates luminance from a spectral power distribution.
Uses the configured CMF and illuminant from ColorScience.
Returns absolute or relative luminance depending on the SPD units.

**Parameters:**
- `spd vec.Vector` – Spectral power distribution values
- `wavelengths vec.Vector` – Wavelength vector (must match spd length)

**Returns:** `(float32, error)` – Luminance (Y component), error

**Example:**
```go
wavelengths := vec.NewFrom(400, 410, 420, 430, 440, 450)
spd := vec.NewFrom(0.5, 0.6, 0.7, 0.65, 0.55, 0.5)
luminance, err := cs.LuminanceFromSPD(spd, wavelengths)
```

#### `CalibrateSensorSensitivity(referenceSPD, sensorResponseSPD)`

Calibrates sensor spectral sensitivity using known reference measurements.
Given a reference SPD with known spectral power and sensor responses to that SPD,
calculates the sensor's spectral sensitivity (responsivity) function.

**Parameters:**
- `referenceSPD SPD` – Known reference SPD (e.g., standard illuminant or calibration lamp)
- `sensorResponseSPD SPD` – Measured sensor response to the reference SPD (must have same wavelengths)

**Returns:** `(SPD, error)` – Calibrated sensitivity SPD (wavelengths, sensitivity values)

**Calibration Process:**
1. Measures sensor response to known reference SPD (calibration lamp, standard illuminant)
2. Calculates sensitivity = response / reference_SPD at each wavelength
3. Returns calibrated sensitivity SPD (wavelengths, sensitivity values)

**Example:**
```go
// Reference: D65 illuminant with known spectral power
d65Ref, _ := LoadIlluminantSPD("D65")
// Sensor response: measured values when exposed to D65
sensorSensitivity, err := CalibrateSensorSensitivity(d65Ref, sensorResponseSPD)
// Result: sensor sensitivity function (responsivity at each wavelength)
```

#### `CalibrateSensorSensitivityWithDark(referenceSPD, sensorResponseSPD, darkSPD)`

Calibrates sensor spectral sensitivity accounting for dark current.
Similar to `CalibrateSensorSensitivity` but subtracts dark current from sensor response first.

**Parameters:**
- `referenceSPD SPD` – Known reference SPD (e.g., standard illuminant or calibration lamp)
- `sensorResponseSPD SPD` – Measured sensor response to the reference SPD
- `darkSPD SPD` – Dark current measurement (sensor response with no light)

**Returns:** `(SPD, error)` – Calibrated sensitivity SPD

**Calibration Process:**
1. Subtracts dark current: corrected_response = response - dark
2. Calculates sensitivity = corrected_response / reference_SPD at each wavelength
3. Returns calibrated sensitivity SPD

**Example:**
```go
d65Ref, _ := LoadIlluminantSPD("D65")
sensorSensitivity, err := CalibrateSensorSensitivityWithDark(d65Ref, sensorResponseSPD, darkSPD)
```

### Color Space Conversions

Color space conversions are available both as standalone functions (returning individual components) and as type methods (using XYZ, LAB, RGB types).

#### Type Methods (Recommended)

**XYZ Methods:**
- `(xyz XYZ) ToLAB(illuminant)` – Convert XYZ to LAB
- `(xyz XYZ) ToRGB(out255)` – Convert XYZ to RGB
- `(xyz XYZ) Luminance()` – Get luminance (Y component)

**LAB Methods:**
- `(lab LAB) ToXYZ(illuminant)` – Convert LAB to XYZ
- `(lab LAB) ToRGB(illuminant, out255)` – Convert LAB to RGB

**RGB Methods:**
- `(rgb RGB) ToXYZ()` – Convert RGB to XYZ
- `(rgb RGB) ToLAB(illuminant)` – Convert RGB to LAB

**Example:**
```go
xyz := NewXYZ(95.0, 100.0, 108.88)
lab := xyz.ToLAB(WhitePointD65_10)
rgb := xyz.ToRGB(true) // 0-255 range

// Chain conversions
rgb := NewRGB(255, 128, 64)
xyz := rgb.ToXYZ()
lab := rgb.ToLAB(WhitePointD65_10)

// Alternative: use xyz
lab2 := xyz.ToLAB(WhitePointD65_10)
rgb2 := lab.ToRGB(WhitePointD65_10, true)
```

#### Standalone Functions

All standalone functions are **not methods on `ColorScience`**.

#### `XYZToLAB(X, Y, Z, illuminant)`

Converts CIE XYZ to CIE LAB color space.

**Parameters:**
- `X, Y, Z float32` – XYZ tristimulus values
- `illuminant WhitePoint` – Reference white point

**Returns:** `(L, a, b float32)`

**Formula:**
- `L = 116 * f(Y/Yn) - 16`
- `a = 500 * (f(X/Xn) - f(Y/Yn))`
- `b = 200 * (f(Y/Yn) - f(Z/Zn))`
- `f(t) = t^(1/3)` if `t > (6/29)^3`, else `t/(3*(6/29)^2) + 4/29`

**Example:**
```go
L, a, b := XYZToLAB(95.0, 100.0, 108.88, WhitePointD65_10)
```

#### `LABToXYZ(L, a, b, illuminant)`

Converts CIE LAB to CIE XYZ color space.

**Parameters:**
- `L, a, b float32` – LAB color values
- `illuminant WhitePoint` – Reference white point

**Returns:** `(X, Y, Z float32)`

#### `XYZToRGB(X, Y, Z, out255)`

Converts CIE XYZ to sRGB.

**Parameters:**
- `X, Y, Z float32` – XYZ tristimulus values (D65 white point assumed)
- `out255 bool` – If true, return 0-255 range; if false, return 0-1 range

**Returns:** `(r, g, b float32)`

**Transform:**
1. Linear RGB = M_XYZ_to_RGB * XYZ
2. Apply sRGB gamma correction
3. Clip to [0, 1] then scale to [0, 255] if `out255=true`

#### `RGBToXYZ(r, g, b)`

Converts sRGB to CIE XYZ (D65 white point).

**Parameters:**
- `r, g, b float32` – RGB values (0-255 or 0-1, auto-detected)

**Returns:** `(X, Y, Z float32)`

**Transform:**
1. Normalize to [0, 1] if values > 1
2. Apply inverse sRGB gamma
3. XYZ = M_RGB_to_XYZ * linear RGB

#### `RGBToLAB(r, g, b, illuminant)`

Chain conversion: RGB → XYZ → LAB.

**Returns:** `(L, a, b float32)`

#### `LABToRGB(L, a, b, illuminant, out255)`

Chain conversion: LAB → XYZ → RGB.

**Returns:** `(r, g, b float32)`

### Chromatic Adaptation

Chromatic adaptation converts XYZ values between different white points.

#### `(xyz XYZ) Adapt(Ws, Wd, method)`

Adapts XYZ values from source white point to destination white point.
Returns adapted XYZ using the specified adaptation method.

**Parameters:**
- `Ws, Wd WhitePoint` – Source and destination white points
- `method AdaptationMethod` – Adaptation method (`AdaptationBradford`, `AdaptationVonKries`, `AdaptationCAT02`)

**Returns:** `(XYZ, error)`

**Example:**
```go
xyz := NewXYZ(95.0, 100.0, 108.88)
xyzAdapted, err := xyz.Adapt(WhitePointD50_10, WhitePointD65_10, AdaptationBradford)
```

#### `AdaptXYZ(X, Y, Z, Ws, Wd, method)`

Adapts XYZ values from source white point to destination white point.
Standalone function returning individual components.

**Parameters:**
- `X, Y, Z float32` – Source XYZ values
- `Ws, Wd WhitePoint` – Source and destination white points
- `method AdaptationMethod` – Adaptation method (`AdaptationBradford`, `AdaptationVonKries`, `AdaptationCAT02`)

**Returns:** `(Xa, Ya, Za float32, error)`

**Algorithm:**
1. Transform white points to RGB using adaptation matrix: `RGB = M * XYZ`
2. Calculate scaling factors: `scale = RGB_dest / RGB_src`
3. Transform source XYZ to RGB: `RGB_src = M * XYZ`
4. Apply scaling: `RGB_adapted = RGB_src * scale`
5. Transform back to XYZ: `XYZ_adapted = M^(-1) * RGB_adapted`

**Example:**
```go
Xa, Ya, Za, err := AdaptXYZ(95.0, 100.0, 108.88, WhitePointD50_10, WhitePointD65_10, AdaptationBradford)
```

### Spectrum Calibration

#### `(spd SPD) Calibrate(pairs ...float32)`

Calibrates SPD wavelengths using known (index, wavelength) pairs with cubic Catmull-Rom spline interpolation.

**Parameters:**
- `pairs ...float32` – Variadic float32 values (must be even length), interpreted as (index, wavelength) pairs

**Returns:** `(SPD, error)`

**Calibration Process:**
1. Parses calibration points from pairs: `(index0, wavelength0, index1, wavelength1, ...)`
2. Validates indices are in range `[0, spd.Len())` and no duplicates
3. Sorts calibration points by index
4. Uses cubic Catmull-Rom spline interpolation between calibration points
5. Uses linear extrapolation for indices before first and after last calibration point
6. Returns new SPD with calibrated wavelengths and original values

**Requirements:**
- At least 2 calibration points required
- SPD must already be initialized with wavelengths and values
- Values row remains unchanged (values stay at their indices, wavelengths are corrected)

**Example:**
```go
initialWl := vec.NewFrom(0.0, 0.0, 0.0, 0.0, 0.0) // Unknown wavelengths
values := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)
spd := NewSPD(initialWl, values)

// Calibrate: index 0 = 400nm, index 2 = 500nm, index 4 = 600nm
calibrated, err := spd.Calibrate(0, 400.0, 2, 500.0, 4, 600.0)
// Wavelengths at indices 1 and 3 are interpolated using cubic spline
```

#### `(cs *ColorScience) Calibrate(dst *SPD, measurement SPD, optWhitePoint ...WhitePoint)`

Calibrates a measurement SPD according to dark and light calibration readings. Normalizes SPD readings to the light SPD.

**Parameters:**
- `dst *SPD` – Output SPD (must already be initialized with wavelengths)
- `measurement SPD` – Raw measurement SPD
- `optWhitePoint ...WhitePoint` – Optional white point for reflective calibration (reserved for future enhancements)

**Returns:** `error`

**Calibration Process:**
1. Validates dark and light calibration SPDs are set
2. Interpolates all SPDs (measurement, dark, light) to same wavelength grid (dst's wavelengths)
3. Subtracts dark current: `corrected_measurement = measurement - dark`, `corrected_light = light - dark`
4. Normalizes to light SPD: `calibrated = corrected_measurement / corrected_light`
5. Stores calibrated values in dst SPD (reflectance/transmittance [0,1] relative to reference light)

**Requirements:**
- Dark and light SPDs must be set via `WithDark` and `WithLight`
- Does not calibrate if dark/light SPDs are not set (returns error)

**Example:**
```go
cs, _ := New(
    WithIlluminant("D65"),
    WithDark(darkSPD),
    WithLight(lightSPD),
)

targetWl := vec.NewFrom(400, 410, 420, /* ... */)
zeroVals := vec.New(targetWl.Len())
dst := NewSPD(targetWl, zeroVals)

err := cs.Calibrate(&dst, rawMeasurementSPD)
// dst now contains normalized values [0,1] relative to light SPD
```

#### `FindWavelengthIndex(wavelengths, targetWavelength)`

Finds the index corresponding to a target wavelength (or nearest).

**Parameters:**
- `wavelengths vec.Vector` – Calibrated wavelength vector
- `targetWavelength float32` – Target wavelength in nanometers

**Returns:** `(index int, exact bool)`

### Spectral Analysis

#### `(spd SPD) Peaks(threshold, minProminence)`

Detects local maxima (peaks) in an SPD.

**Parameters:**
- `threshold float32` – Minimum peak value to consider (0 = no threshold)
- `minProminence float32` – Minimum prominence for a peak to be included (0 = no prominence filter)

**Returns:** `[]Peak` – Slice of detected peaks, sorted by wavelength

**Peak Detection:**
- Detects local maxima where values are higher than their neighbors
- Filters by threshold (minimum value) and prominence (height above surrounding baseline)
- Returns peak index, wavelength, value, and prominence

**Example:**
```go
peaks := spd.Peaks(0.5, 0.1) // Peaks with value >= 0.5 and prominence >= 0.1
for _, peak := range peaks {
    fmt.Printf("Peak at %f nm: value=%.3f, prominence=%.3f\n", 
        peak.Wavelength, peak.Value, peak.Prominence)
}
```

#### `(spd SPD) Valleys(threshold, minProminence)`

Detects local minima (valleys) in an SPD.

**Parameters:**
- `threshold float32` – Maximum valley value to consider (0 = no threshold, use very large value for no filter)
- `minProminence float32` – Minimum prominence for a valley to be included (0 = no prominence filter)

**Returns:** `[]Valley` – Slice of detected valleys, sorted by wavelength

**Valley Detection:**
- Detects local minima where values are lower than their neighbors
- Filters by threshold (maximum value) and prominence (depth below surrounding baseline)
- Returns valley index, wavelength, value, and prominence

#### `(spd SPD) DetectCalibrationPoints(referenceSPD, minConfidence)`

Matches this SPD to a reference SPD and returns calibration points.
Uses correlation-based matching to find corresponding features between the two spectra.
Returns (index, wavelength) pairs that can be used with `SPD.Calibrate()`.

**Parameters:**
- `referenceSPD SPD` – The reference SPD with known wavelengths (e.g., D65 illuminant, calibration target, any known spectrum)
- `minConfidence float32` – Minimum confidence score (0-1) for a match to be included

**Returns:** `[]CalibrationPoint` – Slice of calibration points sorted by index in this SPD

**Algorithm:**
1. Interpolates both SPDs to a common wavelength grid
2. Detects peaks in both spectra for feature matching
3. Uses cross-correlation around peak locations to match peaks
4. Falls back to sliding window correlation if peaks are not found
5. Returns calibration points with confidence scores >= minConfidence

**Example:**
```go
measured := NewSPD(unknownWl, measuredValues)
referenceSPD, _ := LoadIlluminantSPD("D65") // Or any other reference SPD
calPoints := measured.DetectCalibrationPoints(referenceSPD, 0.8)

// Use detected points for calibration
if len(calPoints) >= 2 {
    pairs := make([]float32, 0, len(calPoints)*2)
    for _, cp := range calPoints {
        pairs = append(pairs, float32(cp.Index), cp.Wavelength)
    }
    calibrated, _ := measured.Calibrate(pairs...)
}
```

**Note:** A standalone `DetectCalibrationPoints(measuredSPD, referenceSPD, minConfidence)` function is also available for backward compatibility but is deprecated. Use the method form instead.

### Spectrum Database

#### `NewSpectrumDatabase()`

Creates a new in-memory spectrum database.

**Returns:** `SpectrumDatabase` – New database instance

#### `(db *InMemoryDatabase) Add(name, spd, metadata)`

Adds a spectrum to the database.

**Parameters:**
- `name string` – Spectrum name/identifier
- `spd SPD` – The spectrum SPD
- `metadata SpectrumMetadata` – Metadata about the spectrum

**Returns:** `error`

#### `(db *InMemoryDatabase) SearchBySimilarity(query, maxResults)`

Searches for spectra similar to the given SPD using correlation.

**Parameters:**
- `query SPD` – Query spectrum to match against
- `maxResults int` – Maximum number of results to return (0 = no limit)

**Returns:** `([]MatchResult, error)` – Ranked matches sorted by similarity score (highest first)

#### `(db *InMemoryDatabase) SearchByPeaks(peaks, maxResults)`

Searches for spectra with similar peak patterns.

**Parameters:**
- `peaks []Peak` – Query peaks to match against
- `maxResults int` – Maximum number of results to return (0 = no limit)

**Returns:** `([]MatchResult, error)` – Ranked matches sorted by similarity score (highest first)

**Example:**
```go
db := NewSpectrumDatabase()

// Add spectra to database
d65, _ := LoadIlluminantSPD("D65")
db.Add("D65", d65, SpectrumMetadata{
    Name: "D65",
    Type: "illuminant",
    Description: "CIE Standard Illuminant D65",
})

// Search by similarity
results, _ := db.SearchBySimilarity(querySPD, 10)
for _, result := range results {
    fmt.Printf("Match: %s (score=%.3f, confidence=%.3f)\n",
        result.Entry.Metadata.Name, result.Score, result.Confidence)
}

// Search by peaks
peaks := querySPD.Peaks(0, 0)
results, _ := db.SearchByPeaks(peaks, 10)
```

### Multi-Sensor SPD Reconstruction

#### `(spd SPD) Reconstruct(sensorResponses, useDampedLS, lambda)`

Reconstructs SPD values from multiple photodetector measurements. Modifies the SPD in place (fills values row).

**Parameters:**
- `sensorResponses []SensorResponse` – Array of sensor responses with known spectral sensitivities
- `useDampedLS bool` – If true, use damped least squares (Tikhonov regularization)
- `lambda float32` – Regularization parameter (only used if `useDampedLS=true`)

**Returns:** `error`

**Requirements:**
- SPD must already be initialized with wavelengths row and zeroed values row

**Algorithm:**
1. Build sensor response matrix `S` (num_sensors × num_wavelengths)
2. Each row is a sensor's spectral response interpolated to target wavelengths
3. Solve linear system: `R = S · X` where:
   - `R` = sensor readings vector
   - `S` = sensor response matrix
   - `X` = unknown stimulus SPD
4. Solution methods:
   - Standard: `X = S^+ · R` (pseudo-inverse)
   - Damped LS: `X = (S^T·S + λ²I)^(-1) · S^T · R`
5. Enforce non-negativity constraint (SPD cannot be negative)
6. Fill values row of SPD with reconstructed values

**Example:**
```go
targetWl := vec.NewFrom(400, 410, 420, /* ... */)
zeroVals := vec.New(targetWl.Len())
spd := NewSPD(targetWl, zeroVals)

sensors := []SensorResponse{
    {Name: "Sensor1", Response: sensor1SPD, Reading: 0.85},
    {Name: "Sensor2", Response: sensor2SPD, Reading: 0.72},
    // ... more sensors
}

err := spd.Reconstruct(sensors, false, 0.01)
// spd now contains reconstructed values
```

#### `ReconstructSPDWithConstraints(sensorResponses, targetWavelengths, useDampedLS, lambda, smoothnessWeight)`

Convenience wrapper that creates SPD with target wavelengths and zeroed values, then calls `SPD.Reconstruct()`.

**Returns:** `(SPD, error)` – Returns reconstructed SPD as value (not pointer)

**Future Enhancements:**
- Smoothness regularization (second derivative penalty)
- Known value constraints
- Bounds on values

### Integration

#### `IntegrationKernel(wavelengths)`

Computes trapezoidal integration kernel for numerical integration.

**Parameters:**
- `wavelengths vec.Vector` – Wavelength vector

**Returns:** `vec.Vector` – Integration kernel

**Formula:**
- First point: `kernel[0] = (wavelengths[1] - wavelengths[0]) / 2`
- Middle points: `kernel[i] = (wavelengths[i+1] - wavelengths[i-1]) / 2`
- Last point: `kernel[n-1] = (wavelengths[n-1] - wavelengths[n-2]) / 2`

**Usage:** `∫f(λ)·dλ ≈ kernel · f(λ)` (dot product)

#### `Integrate(wavelengths, values)`

Computes integral using trapezoidal rule.

**Parameters:**
- `wavelengths vec.Vector` – Wavelength vector
- `values vec.Vector` – Function values

**Returns:** `float32` – Integral value

**Implementation:** `Integrate(wl, vals) = IntegrationKernel(wl).Dot(vals)`

## Embedded Data

The following data files are embedded at compile time via `embed` package:

### Observers
- `data/CIE_xyz_1931_2deg.csv` – CIE 1931 2 Degree Standard Observer (x̄, ȳ, z̄)
- `data/CIE_xyz_1964_10deg.csv` – CIE 1964 10 Degree Standard Observer (x̄, ȳ, z̄)

### Illuminants
- `data/CIE_std_illum_A_1nm.csv` – Standard Illuminant A (1nm resolution)
- `data/CIE_std_illum_D50.csv` – Standard Illuminant D50
- `data/CIE_std_illum_D65.csv` – Standard Illuminant D65

**Data Format:** CSV files with `wavelength,value1,value2,...` format. Observer files have 3 value columns (x̄, ȳ, z̄); illuminant files have 1 value column.

## White Points

Standard white point constants are provided:

### D Illuminants (10° observer)
- `WhitePointD50_10` – (96.72, 100.000, 81.43)
- `WhitePointD55_10` – (95.682, 100.000, 92.149)
- `WhitePointD65_10` – (94.81, 100.000, 107.32) – **Default**
- `WhitePointD75_10` – (94.972, 100.000, 122.638)

### D Illuminants (2° observer)
- `WhitePointD50_2` – (96.422, 100.000, 82.521)
- `WhitePointD65_2` – (95.047, 100.000, 108.883)
- `WhitePointD75_2` – (94.972, 100.000, 122.638)

### Standard Illuminants
- `WhitePointA` – (109.850, 100.000, 35.585)
- `WhitePointB` – (99.092, 100.000, 85.313)
- `WhitePointC` – (98.074, 100.000, 118.232)
- `WhitePointE` – (100.000, 100.000, 100.000)

### F-Series Illuminants
- `WhitePointF1` through `WhitePointF12`

### LED White Points
- `WhitePointLED_CW_6500K` – (95.04, 100.0, 108.88)
- `WhitePointLED_NW_4300K` – (97.0, 100.0, 92.0)
- `WhitePointLED_WW_3000K` – (98.5, 100.0, 67.0)
- `WhitePointLED_VWW_2200K` – (103.0, 100.0, 50.0)

## Error Handling

### Common Errors

1. **Nil Parameters**
   - `New()` with nil CMF or illuminant returns error: `"CMF cannot be nil"` or `"illuminant SPD cannot be nil"`

2. **Length Mismatches**
   - `ComputeXYZ()` with mismatched spd/wavelengths lengths returns: `"wavelengths (%d) and spd (%d) lengths must match"`

3. **Invalid Illuminant**
   - `LoadIlluminantSPD()` with unknown name returns: `"unknown illuminant '%s'. Available: %v"`

4. **Invalid Observer**
   - `LoadCMF()` with invalid observer returns: `"unknown observer type: %s (use '2' or '10')"`

5. **Calibration Errors**
   - `CalibrateSpectrum()` with empty mapping returns: `"mapping cannot be empty"`
   - Out-of-range indices are silently discarded (no error)

6. **Reconstruction Errors**
   - `ReconstructSPD()` with no sensors returns: `"at least one sensor response required"`
   - Pseudo-inverse failure returns: `"failed to compute pseudo-inverse: %w"`

## Performance Considerations

1. **Integration Kernel** – Uses vector dot product instead of manual loops for efficiency
2. **SPD Interpolation** – Linear interpolation is O(n) where n is target wavelength count
3. **CMF/Illuminant Interpolation** – Performed on-demand when wavelengths don't match; consider caching if same wavelengths are used repeatedly
4. **Multi-Sensor Reconstruction** – Pseudo-inverse is O(min(m,n)²) where m=num_sensors, n=num_wavelengths. Use damped LS for ill-conditioned systems.
5. **Embedded Data** – Loaded once at package initialization; no runtime file I/O

## Testing

Comprehensive unit tests cover:

- Construction (`New`, options)
- Data loading (CMF, illuminants, error cases)
- Spectrum-to-XYZ conversion
- Color space conversions (round-trip tests)
- Chromatic adaptation
- Spectrum calibration (index mapping, out-of-range handling)
- Integration kernel and numerical integration
- Error cases (nil inputs, length mismatches, invalid parameters)

**Test File:** `colorscience_test.go`

**Run Tests:**
```bash
go test -v ./x/math/colorscience
```

## Dependencies

- `x/math/vec` – Vector operations
- `x/math/mat` – Matrix operations (for adaptation and reconstruction)
- `x/math/interpolation` – Linear interpolation (`Lerp`)
- `embed` – Embedded data files

## References

- **CIE 15:2018** – Colorimetry, 4th Edition
- **Wyszecki & Stiles (2000)** – Color Science: Concepts and Methods, Quantitative Data and Formulae (2nd ed.). Wiley.
- **sRGB Standard** – IEC 61966-2-1:1999

## Missing Operations

See `MISSING_OPS.md` for documentation of missing vector/matrix operations and implemented workarounds.

## Missing Features & Future Enhancements

### High Priority Missing Features

#### 1. Spectral Characteristics (Future Enhancements)

**Spectral Characteristics:**
- `(spd SPD) DominantWavelength()` – Find dominant wavelength (peak intensity)
- `(spd SPD) CentroidWavelength()` – Calculate spectral centroid
- `(spd SPD) Bandwidth(halfMax)` – Calculate spectral bandwidth (FWHM, etc.)
- `(spd SPD) SpectralFeatures()` – Extract comprehensive feature set

#### 4. Spectroscopy Types Support

**Raman Spectroscopy:**
- `RamanShift(excitationWl, scatteredWl)` – Calculate Raman shift (cm⁻¹)
- `RamanIntensity(scatteredSPD, excitationWl)` – Extract Raman intensity
- `MatchRamanSpectrum(spd SPD, database)` – Raman-specific matching (peak-based)

**Absorption Spectroscopy:**
- `Absorbance(incidentSPD, transmittedSPD)` – Calculate absorbance: `A = -log₁₀(T/I)`
- `Transmittance(incidentSPD, transmittedSPD)` – Calculate transmittance: `T/I`
- `BeerLambertLaw(absorbance, pathLength, concentration)` – Beer-Lambert law calculations
- `ExtinctionCoefficient(absorptivity, pathLength)` – Calculate extinction coefficient

**Reflectance Spectroscopy:**
- Already supported via `ColorScience.Calibrate()` (normalizes to light SPD)
- `ReflectanceFactor(calibratedSPD, referenceWhiteSPD)` – Calculate reflectance factor
- `RemissionFunction(reflectance)` – Calculate remission function (Kubelka-Munk)

**Fluorescence Spectroscopy:**
- `EmissionSpectrum(excitationWl, emissionSPD)` – Process emission spectrum
- `ExcitationSpectrum(emissionWl, excitationSPD)` – Process excitation spectrum
- `QuantumYield(emission, absorption)` – Calculate quantum yield

### Nice-to-Have Features

#### 5. Spectral Processing

**Smoothing & Filtering:**
- `(spd SPD) Smooth(method, windowSize)` – Apply smoothing (Savitzky-Golay, moving average, Gaussian)
- `(spd SPD) Filter(filterType, cutoff)` – Apply spectral filters (low-pass, high-pass, band-pass)
- `(spd SPD) BaselineCorrection(method)` – Baseline correction (polynomial, spline, asymmetric least squares)

**Normalization:**
- `(spd SPD) Normalize(method)` – Normalize SPD (area, max, min-max, L2)
- `(spd SPD) Derivative(order)` – Calculate spectral derivatives (1st, 2nd order)

#### 6. Advanced Calibration

**Multi-Point Calibration:**
- `CalibrateWithReference(spd, referenceSPD, method)` – Calibrate using full reference spectrum
- Support for non-linear calibration (polynomial, spline)
- Uncertainty propagation for calibrated wavelengths

**White Point Calibration:**
- Complete implementation of white point normalization in `Calibrate()`
- Support for reflective calibration targets (normalize to 100% reflectance)

#### 7. Spectral Database & I/O

**Standard Formats:**
- `LoadSPD(format, path)` – Load SPD from various formats (CSV, JSON, ENVI, ASD)
- `SaveSPD(spd SPD, format, path)` – Save SPD to standard formats
- Support for metadata (acquisition parameters, instrument info)

**Database Integration:**
- Integration with external databases (NIST, CIE, material databases)
- Caching and indexing for fast searches
- Support for large-scale spectral libraries

#### 8. Statistical Analysis

**Spectral Statistics:**
- `(spd SPD) Mean()`, `StdDev()`, `Skewness()`, `Kurtosis()` – Statistical moments
- `Compare(spd1, spd2, metric)` – Compare spectra using various metrics (RMSD, correlation, SAM, SID)

**Uncertainty Analysis:**
- Uncertainty propagation in calculations
- Confidence intervals for calibrated wavelengths
- Error estimation in reconstructions

#### 9. Visualization Support

**Spectral Plotting:**
- Export to formats suitable for plotting (CSV, JSON)
- Integration with plotting libraries
- Support for multi-spectrum overlays

#### 10. Performance Optimizations

**Caching:**
- Cache interpolated CMF/illuminant SPDs for common wavelength grids
- Cache correlation matrices for database searches

**Parallel Processing:**
- Parallel search in large databases
- Parallel processing of multiple spectra
- GPU acceleration for large-scale operations

### Implementation Status

#### Phase 1 (✅ Implemented)
- ✅ Peak/valley detection (`Peaks`, `Valleys`)
- ✅ Automatic calibration point detection (`DetectCalibrationPoints`) - matches to any spectrum
- ✅ Basic spectrum database interface (`NewSpectrumDatabase`, `Add`, `SearchBySimilarity`, `SearchByPeaks`)
- ✅ Luminosity calculations (`Luminance`, `LuminanceFromSPD`)
- ✅ Sensor sensitivity calibration (`CalibrateSensorSensitivity`, `CalibrateSensorSensitivityWithDark`)

#### Phase 2 (Short-term - Enhanced Usability)
- Spectral characteristics (DominantWavelength, CentroidWavelength, Bandwidth, SpectralFeatures)
- Raman/Absorption/Reflectance spectroscopy helpers
- White point calibration completion
- Enhanced spectrum matching algorithms

#### Phase 3 (Medium-term - Advanced Features)
- Full spectrum database with persistence (load/save from files)
- Spectral processing (smoothing, filtering, baseline correction)
- Standard format I/O support (CSV, JSON, ENVI, ASD)

#### Phase 4 (Long-term - Research/Enterprise)
- External database integration (NIST, CIE, material databases)
- Uncertainty analysis and error propagation
- GPU acceleration for large-scale operations
- Large-scale spectral library support

## Package Structure

```
colorscience/
├── colorscience.go      # Main ColorScience struct and ComputeXYZ
├── conversions.go       # Standalone color space conversions
├── color_types.go       # XYZ, LAB, RGB types and conversion helpers
├── adaptation.go        # Chromatic adaptation transforms
├── calibration.go       # Spectrum calibration (index mapping, FindWavelengthIndex)
├── reconstruction.go    # Multi-sensor SPD reconstruction
├── integration.go       # Integration kernel and numerical integration
├── spd.go              # SPD type and interpolation, wavelength calibration
├── analysis.go         # Spectral analysis (Peaks, Valleys, DetectCalibrationPoints)
├── database.go         # Spectrum database interface and implementation
├── luminance.go        # Luminosity calculations and sensor sensitivity calibration
├── data_loader.go      # DataLoader for embedded CSV files
├── loader.go           # Standalone loader functions (LoadCMF, LoadIlluminantSPD)
├── options.go          # Options pattern for New()
├── white_points.go     # White point constants
├── data/               # Embedded CSV data files
│   ├── CIE_xyz_1931_2deg.csv
│   ├── CIE_xyz_1964_10deg.csv
│   ├── CIE_std_illum_A_1nm.csv
│   ├── CIE_std_illum_D50.csv
│   └── CIE_std_illum_D65.csv
├── colorscience_test.go # Unit tests
├── SPEC.md             # This file
└── MISSING_OPS.md      # Missing operations documentation
```

