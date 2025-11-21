# AS734x: Obtaining 1nm Resolution SPD

This guide outlines the steps needed to obtain a Spectral Power Distribution (SPD) with 1nm resolution from AS734x sensors, assuming factory-calibrated filter characteristics.

## Overview

AS734x sensors provide **8-18 channel measurements** (not 1nm resolution), so **spectral reconstruction** is required to interpolate/extrapolate to 1nm resolution. The process involves:

1. **Dark Current Measurement** - Required (offset correction)
2. **Filter Response Specification** - Use datasheet values (factory-calibrated assumption)
3. **Raw Measurement** - Read sensor channels
4. **Corrections** - Apply dark current correction
5. **Spectral Reconstruction** - Reconstruct full 1nm spectrum from channel measurements

## Step-by-Step Guide

### Prerequisites

- AS734x device configured and connected
- Access to `colorscience` package for spectral reconstruction
- Dark environment (for dark current measurement)

### Step 1: Measure Dark Current (Required)

Even with factory-calibrated filters, **dark current/offset must be measured** for each device as it varies with:
- Temperature
- Integration time
- Gain setting
- Device-to-device variation

```go
import (
    "time"
    "github.com/itohio/EasyRobot/x/devices/as734x"
    "github.com/itohio/EasyRobot/x/devices"
)

// 1. Initialize device
bus := devices.NewI2C(...) // Your I2C bus
dev := as734x.New(bus, as734x.DefaultAddress)

cfg := as734x.DefaultConfig()
cfg.Gain = as734x.Gain16x
cfg.ATime = 29  // Integration time parameter
cfg.AStep = 599

err := dev.Configure(cfg)
if err != nil {
    log.Fatal(err)
}

// 2. Measure dark current (sensor must be covered/shuttered)
// Use the SAME integration time and gain as your measurements
darkCal := dev.MeasureDarkCurrent(
    100,                          // Number of samples to average
    100*time.Millisecond,        // Integration time (adjust to match measurements)
    as734x.Gain16x,              // Gain (adjust to match measurements)
    25.0,                        // Ambient temperature (optional, 0 = not measured)
)
```

**Important Notes:**
- Dark current should be **remeasured** if:
  - Temperature changes significantly (>5°C)
  - Integration time or gain changes
  - Device has aged significantly
- Store dark current calibration with measurement conditions

### Step 2: Define Filter Response Characteristics (From Datasheet)

Use factory-calibrated filter specifications from the datasheet. AS734x uses interference filters with known center wavelengths and FWHM (Full Width at Half Maximum).

```go
import (
    "github.com/itohio/EasyRobot/x/math/colorscience"
    "github.com/itohio/EasyRobot/x/math/vec"
)

// AS7341 has 8 visible channels + Clear + NIR
func GetAS7341ChannelSpecs() []colorscience.SensorChannel {
    return []colorscience.SensorChannel{
        {Name: "F1", CenterWL: 415, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F2", CenterWL: 445, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F3", CenterWL: 480, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F4", CenterWL: 515, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F5", CenterWL: 555, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F6", CenterWL: 590, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F7", CenterWL: 630, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F8", CenterWL: 680, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        // Note: Clear channel has no filter, NIR channel typically not used for visible reconstruction
    }
}

// Create target wavelength vector (350-1000nm, 1nm steps)
wavelengths := vec.New(651) // 1000 - 350 + 1 = 651 wavelengths
for i := 0; i < 651; i++ {
    wavelengths[i] = float32(350 + i)
}
```

**Key Points:**
- Use **Super-Gaussian n=4** filter shape (best matches interference filters)
- Center wavelengths and FWHM from datasheet are typically accurate enough
- If needed, you can measure actual filter responses with a monochromator (advanced)

### Step 3: Take Raw Measurement

```go
// Take a measurement
raw, err := dev.Read()
if err != nil {
    log.Fatal(err)
}

// raw.Channels contains channel readings (8-18 values depending on variant)
// raw.IntegrationUs, raw.Gain contain measurement conditions
```

### Step 4: Apply Dark Current Correction

```go
// Apply dark current correction
correctedChannels := make([]float32, len(raw.Channels))
for i, rawValue := range raw.Channels {
    correctedChannels[i] = float32(rawValue) - darkCal.Offsets[i]
    if correctedChannels[i] < 0 {
        correctedChannels[i] = 0 // Non-negative constraint
    }
}
```

### Step 5: Reconstruct Full Spectrum (1nm Resolution)

Reconstruct full spectrum from corrected channel measurements using the `colorscience` package.

```go
import (
    "github.com/itohio/EasyRobot/x/math/colorscience"
)

// Prepare channel data for reconstruction
channels := GetAS7341ChannelSpecs()

// Set corrected readings for each channel
for i := 0; i < len(channels) && i < len(correctedChannels); i++ {
    channels[i].Reading = correctedChannels[i]
}

// Reconstruct full spectrum (1nm resolution)
spd, err := colorscience.ReconstructSPDFromChannels(
    channels,           // Channel specifications with readings
    wavelengths,        // Target wavelengths (1nm steps)
    true,              // useDampedLS = true (recommended for stability)
    0.01,              // lambda = 0.01 (regularization parameter)
)
if err != nil {
    log.Fatal(err)
}

// spd now contains full spectrum with 1nm resolution (350-1000nm)
// Access values:
spdWavelengths := spd.Wavelengths()    // Wavelength vector
spdValues := spd.Values()              // Spectral values at each wavelength
```

### Complete Example: Simple Workflow

```go
package main

import (
    "fmt"
    "log"
    "time"
    
    "github.com/itohio/EasyRobot/x/devices/as734x"
    "github.com/itohio/EasyRobot/x/devices"
    "github.com/itohio/EasyRobot/x/math/colorscience"
    "github.com/itohio/EasyRobot/x/math/vec"
)

func main() {
    // Initialize device
    bus := devices.NewI2C(...) // Your I2C implementation
    dev := as734x.New(bus, as734x.DefaultAddress)
    
    cfg := as734x.DefaultConfig()
    cfg.Gain = as734x.Gain16x
    cfg.ATime = 29
    cfg.AStep = 599
    
    if err := dev.Configure(cfg); err != nil {
        log.Fatal(err)
    }
    
    // Step 1: Measure dark current (cover sensor!)
    fmt.Println("Measuring dark current... (cover sensor)")
    darkCal := dev.MeasureDarkCurrent(100, 100*time.Millisecond, as734x.Gain16x, 25.0)
    
    // Step 2: Define channel specs and target wavelengths
    channels := GetAS7341ChannelSpecs()
    wavelengths := createWavelengthVector(350, 1000, 1) // 1nm steps
    
    // Step 3: Take measurement
    fmt.Println("Taking measurement...")
    raw, err := dev.Read()
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 4: Apply dark current correction
    correctedChannels := applyDarkCorrection(raw.Channels, darkCal.Offsets)
    
    // Step 5: Reconstruct full spectrum
    for i := 0; i < len(channels) && i < len(correctedChannels); i++ {
        channels[i].Reading = correctedChannels[i]
    }
    
    spd, err := colorscience.ReconstructSPDFromChannels(
        channels, wavelengths, true, 0.01,
    )
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 6: Use reconstructed spectrum
    fmt.Printf("Reconstructed spectrum: %d wavelengths\n", spd.Wavelengths().Len())
    
    // Example: Get value at specific wavelength
    targetWL := float32(550.0)
    spdValues := spd.Values()
    spdWavelengths := spd.Wavelengths()
    
    // Find closest wavelength or interpolate
    for i := 0; i < spdWavelengths.Len(); i++ {
        if spdWavelengths[i] >= targetWL {
            fmt.Printf("Value at %.1fnm: %.4f\n", targetWL, spdValues[i])
            break
        }
    }
}

func GetAS7341ChannelSpecs() []colorscience.SensorChannel {
    return []colorscience.SensorChannel{
        {Name: "F1", CenterWL: 415, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F2", CenterWL: 445, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F3", CenterWL: 480, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F4", CenterWL: 515, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F5", CenterWL: 555, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F6", CenterWL: 590, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F7", CenterWL: 630, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
        {Name: "F8", CenterWL: 680, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
    }
}

func createWavelengthVector(start, end, step int) vec.Vector {
    n := (end - start) / step + 1
    wl := vec.New(n)
    for i := 0; i < n; i++ {
        wl[i] = float32(start + i*step)
    }
    return wl
}

func applyDarkCorrection(rawChannels []uint16, offsets []float32) []float32 {
    corrected := make([]float32, len(rawChannels))
    for i := range rawChannels {
        corrected[i] = float32(rawChannels[i]) - offsets[i]
        if corrected[i] < 0 {
            corrected[i] = 0
        }
    }
    return corrected
}
```

## What Accuracy to Expect?

### With Factory-Calibrated Filter Specs Only:

**Typical Accuracy:**
- **Colorimetry (XYZ, LAB)**: ±2-5% accuracy (acceptable for most color measurement applications)
- **Spectral Shape**: Good for general spectral shape, moderate accuracy at specific wavelengths
- **Absolute Spectral Power**: Moderate accuracy (±10-20%) without absolute calibration

### Limitations:

1. **Filter Overlap**: Channels have overlapping responses (especially adjacent channels)
   - Reconstruction handles this mathematically, but accuracy decreases at overlap regions

2. **Out-of-Band Transmission**: Filters have some response outside their nominal band
   - Less critical for visible spectrum reconstruction

3. **Device-to-Device Variation**: While filters are factory-calibrated, there is still some variation
   - Typically ±2-5nm wavelength shift possible

4. **Reconstruction Uncertainty**: Going from 8-18 channels to 651 points (1nm resolution) involves interpolation/extrapolation
   - More uncertainty at wavelengths far from channel centers
   - Better accuracy near channel center wavelengths

### Improving Accuracy (Optional):

1. **Matrix-Based Calibration** (Best accuracy):
   - Measure test targets (RGBWCMY patches) with both AS734x and reference spectrometer
   - Calibrate reconstruction matrix from paired measurements
   - **Accuracy**: Can achieve ±1-2% for colorimetry, ±5-10% for absolute spectral power

2. **Individual Filter Response Measurement**:
   - Measure each channel's spectral response with monochromator
   - Use measured responses instead of datasheet specs
   - **Accuracy**: Improved spectral shape accuracy

3. **Absolute Calibration**:
   - Calibrate against known reference illuminant (e.g., D65 lamp)
   - Scale to absolute spectral irradiance
   - **Accuracy**: Enables absolute power measurements

## Quick Start Summary

**Minimum Required Steps for 1nm SPD:**

1. ✅ **Measure dark current** (covered sensor, ~100 samples)
2. ✅ **Use datasheet filter specs** (center WL, FWHM, Super-Gaussian n=4)
3. ✅ **Take raw measurement** (`dev.Read()`)
4. ✅ **Apply dark correction** (subtract offsets)
5. ✅ **Reconstruct spectrum** (`ReconstructSPDFromChannels()` with 1nm wavelength vector)

**Result:** Full spectrum with 1nm resolution (350-1000nm), suitable for colorimetry and general spectral analysis.

**Expected Accuracy:** ±2-5% for colorimetry, moderate for absolute spectral power.

## Notes

- **Dark current** must be remeasured periodically (temperature-dependent)
- **Filter specs** from datasheet are sufficient assuming factory calibration
- **Reconstruction method**: Use damped least squares (`useDampedLS=true, lambda=0.01`) for stability
- **Wavelength range**: Typically 350-1000nm, but AS7341 is most accurate 400-700nm (visible)
- **Integration time and gain**: Match settings between dark current measurement and actual measurements

## See Also

- `calib.md` - Complete calibration API documentation
- `colorscience` package - Spectral reconstruction functions
- [ams Application Note AN000633](https://look.ams-osram.com/m/269928fe0dba7511/original/Spectral-Sensor-Calibration-Methods.pdf) - Official calibration methods
