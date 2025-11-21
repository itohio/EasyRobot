# AS734x Spectral Sensor Calibration API

This document describes the calibration API for AS734x spectral sensors based on the [ams Application Note AN000633: Spectral Sensor Calibration Methods](https://look.ams-osram.com/m/269928fe0dba7511/original/Spectral-Sensor-Calibration-Methods.pdf).

## Overview

The AS734x sensors require multiple calibration steps to convert raw sensor counts into accurate spectral data. The calibration process corrects for:

1. **Filter Response**: Non-ideal filter characteristics, overlap, and out-of-band transmission
2. **Diffuser Effects**: Diffuser transmission characteristics if a diffuser is used
3. **Dark Current**: Sensor dark current and offset
4. **NIR Correction**: NIR channel contamination in visible channels
5. **Normalization**: Scaling to physical units and reference illuminants
6. **Spectral Reconstruction**: Matrix-based reconstruction of full spectrum from channel measurements

## Calibration Data Structures

### CalibrationData

Contains all calibration parameters for a device.

```go
type CalibrationData struct {
    // Device identification
    DeviceSerial string    // Device serial number or identifier
    Variant      Variant   // AS7341 or AS7343
    CalibratedAt time.Time // Timestamp of calibration
    
    // Offset/Dark current correction
    DarkCurrent DarkCurrentCalibration
    
    // Filter response correction
    FilterResponse FilterResponseCalibration
    
    // Diffuser compensation (if applicable)
    Diffuser *DiffuserCalibration // nil if no diffuser
    
    // NIR correction
    NIRCorrection NIRCalibration
    
    // Normalization/scale factors
    Normalization NormalizationCalibration
    
    // Matrix-based spectral reconstruction
    ReconstructionMatrix *ReconstructionMatrix // nil if not calibrated
    
    // Metadata
    CalibrationMethod string // "manual", "reference_spectrometer", "matrix", etc.
    ReferenceDevice   string // Reference spectrometer used (if applicable)
}
```

### DarkCurrentCalibration

Calibration data for dark current/offset correction.

```go
type DarkCurrentCalibration struct {
    // Per-channel dark current values (in raw counts)
    Offsets []float32 // Length: num_channels (10 for AS7341, 18 for AS7343)
    
    // Temperature-dependent correction (optional)
    TemperatureCoefficient []float32 // Per-channel temp coefficient
    ReferenceTemperature   float32   // Temperature at which offsets were measured
    
    // Integration time and gain at which offsets were measured
    IntegrationTime time.Duration
    Gain            Gain
}
```

### FilterResponseCalibration

Calibration data for filter response correction. This can include:
- Per-channel correction factors
- Spectral response curves (for matrix-based reconstruction)
- Out-of-band transmission factors

```go
type FilterResponseCalibration struct {
    // Option 1: Simple per-channel correction factors
    CorrectionFactors []float32 // Per-channel multiplicative factors
    
    // Option 2: Spectral response matrices (for advanced reconstruction)
    // Each channel's spectral response as SPD (wavelengths -> response)
    SpectralResponses []colorscience.SPD // One per channel
    
    // Option 3: Center wavelength and FWHM (for colorscience reconstruction)
    ChannelSpecs []colorscience.SensorChannel // CenterWL, FWHM per channel
    
    // Filter overlap correction matrix
    OverlapMatrix *mat.Matrix // Correction for filter overlap (num_channels × num_channels)
    
    // Out-of-band transmission factors
    OutOfBandFactors []float32 // Per-channel out-of-band transmission
}
```

### DiffuserCalibration

Calibration data for diffuser compensation.

```go
type DiffuserCalibration struct {
    // Diffuser transmission spectrum
    Transmission colorscience.SPD // Wavelengths -> transmission (0-1)
    
    // Diffuser type
    Type string // "achromatic", "custom", etc.
    
    // Optional: wavelength-dependent correction factors
    CorrectionFactors colorscience.SPD // Per-wavelength correction
}
```

### NIRCalibration

Calibration data for NIR channel contamination correction.

```go
type NIRCalibration struct {
    // NIR contribution factor to each visible channel
    NIRContribution []float32 // Contribution factor per visible channel
    
    // Correction enabled flag
    Enabled bool
}
```

### NormalizationCalibration

Calibration data for normalization and scaling.

```go
type NormalizationCalibration struct {
    // Reference illuminant used for normalization
    ReferenceIlluminant string // "D65", "D50", "A", "F11", etc.
    
    // Scaling factors per channel (to convert to spectral irradiance/power)
    ScaleFactors []float32 // Per-channel scaling factors
    
    // Integration time and gain for scale factors
    ReferenceIntegrationTime time.Duration
    ReferenceGain            Gain
    
    // White point normalization (for colorimetry)
    WhitePoint colorscience.WhitePoint // CIE XYZ white point
}
```

### ReconstructionMatrix

Matrix-based spectral reconstruction calibration.

```go
type ReconstructionMatrix struct {
    // Reconstruction matrix (num_channels × num_wavelengths)
    Matrix mat.Matrix // Maps channel readings to wavelength spectrum
    
    // Target wavelengths for reconstruction
    Wavelengths vec.Vector // Wavelength vector (nm)
    
    // Method used: "pseudo_inverse", "damped_least_squares", "wiener"
    Method string
    
    // Regularization parameter (for damped LS)
    Lambda float32
    
    // Validation metrics
    RMSE   float32 // Root mean square error
    MaxError float32 // Maximum reconstruction error
}
```

## API Methods

### 1. Dark Current Calibration

#### MeasureDarkCurrent

Measures dark current for all channels with sensor covered/shuttered.

```go
func (d *Device) MeasureDarkCurrent(
    samples int,              // Number of samples to average
    integrationTime time.Duration, // Integration time
    gain Gain,                // Gain setting
    temperature float32,      // Ambient temperature (optional, 0 = not measured)
) (DarkCurrentCalibration, error)

// Usage:
darkCal := d.MeasureDarkCurrent(
    100,              // 100 samples
    100*time.Millisecond, // 100ms integration
    Gain16x,          // 16x gain
    25.0,             // 25°C
)
```

#### ApplyDarkCurrentCorrection

Applies dark current correction to raw measurements.

```go
func (d *Device) ApplyDarkCurrentCorrection(
    raw RawMeasurement,
    darkCal DarkCurrentCalibration,
    temperature float32, // Current temperature (optional, 0 = no temp correction)
) (CorrectedMeasurement, error)

// Usage:
corrected, err := d.ApplyDarkCurrentCorrection(raw, darkCal, 25.0)
```

### 2. Filter Response Calibration

#### LoadFilterResponseFromSpecs

Creates filter response calibration from center wavelength and FWHM specifications.

```go
func LoadFilterResponseFromSpecs(
    variant Variant,
    channelSpecs []colorscience.SensorChannel, // CenterWL, FWHM per channel
    filterShape colorscience.FilterShapeModel, // Gaussian, SuperGaussian4, etc.
    targetWavelengths vec.Vector, // Target wavelengths for response curves
) (FilterResponseCalibration, error)

// Usage:
// AS7341 channels: F1-F8, Clear, NIR
channels := []colorscience.SensorChannel{
    {Name: "F1", CenterWL: 415, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
    {Name: "F2", CenterWL: 445, FWHM: 20, FilterShape: int(colorscience.FilterShapeSuperGaussian4)},
    // ... more channels
}
wavelengths := vec.NewFrom(350.0, 351.0, ..., 1000.0)
filterCal := LoadFilterResponseFromSpecs(VariantAS7341, channels, 
    colorscience.FilterShapeSuperGaussian4, wavelengths)
```

#### LoadFilterResponseFromMeasurements

Creates filter response calibration from measured spectral responses (requires reference spectrometer).

```go
func LoadFilterResponseFromMeasurements(
    variant Variant,
    measuredResponses []colorscience.SPD, // One SPD per channel (from reference spectrometer)
) (FilterResponseCalibration, error)

// Usage:
// Requires reference spectrometer measurements of each channel's spectral response
var channelResponses []colorscience.SPD
for _, channel := range channels {
    // Measure with reference spectrometer
    response := measureChannelWithSpectrometer(channel)
    channelResponses = append(channelResponses, response)
}
filterCal := LoadFilterResponseFromMeasurements(VariantAS7341, channelResponses)
```

#### ApplyFilterCorrection

Applies filter response correction to measurements.

```go
func (d *Device) ApplyFilterCorrection(
    corrected CorrectedMeasurement, // After dark current correction
    filterCal FilterResponseCalibration,
) (CorrectedMeasurement, error)

// Usage:
corrected = d.ApplyFilterCorrection(corrected, filterCal)
```

### 3. Diffuser Compensation

#### LoadDiffuserCalibration

Loads diffuser calibration from transmission spectrum.

```go
func LoadDiffuserCalibration(
    transmission colorscience.SPD, // Wavelengths -> transmission (0-1)
    diffuserType string,           // "achromatic", "custom", etc.
) DiffuserCalibration

// Usage:
// Load diffuser transmission data (measured or from datasheet)
diffuserSPD := colorscience.LoadSPDFromCSV("diffuser_transmission.csv")
diffuserCal := LoadDiffuserCalibration(diffuserSPD, "achromatic")
```

#### ApplyDiffuserCompensation

Applies diffuser compensation to measurements.

```go
func (d *Device) ApplyDiffuserCompensation(
    corrected CorrectedMeasurement,
    diffuserCal DiffuserCalibration,
) (CorrectedMeasurement, error)

// Usage:
if calData.Diffuser != nil {
    corrected = d.ApplyDiffuserCompensation(corrected, *calData.Diffuser)
}
```

### 4. NIR Correction

#### CalibrateNIRCorrection

Calibrates NIR contribution to visible channels using monochromatic light sources.

```go
func (d *Device) CalibrateNIRCorrection(
    nirReading float32,        // NIR channel reading
    visibleReadings []float32, // Visible channel readings
    // ... measurement conditions
) (NIRCalibration, error)
```

#### ApplyNIRCorrection

Applies NIR correction to visible channels.

```go
func (d *Device) ApplyNIRCorrection(
    corrected CorrectedMeasurement,
    nirCal NIRCalibration,
) (CorrectedMeasurement, error)

// Usage:
corrected = d.ApplyNIRCorrection(corrected, calData.NIRCorrection)
```

### 5. Normalization

#### CalibrateNormalization

Calibrates normalization factors using a reference illuminant.

```go
func CalibrateNormalization(
    corrected CorrectedMeasurement,   // Corrected channel readings
    referenceSPD colorscience.SPD,    // Reference illuminant SPD (e.g., D65)
    targetIlluminant string,          // Target illuminant name
    integrationTime time.Duration,    // Integration time used
    gain Gain,                        // Gain used
) (NormalizationCalibration, error)

// Usage:
d65SPD := colorscience.LoadIlluminantSPD("D65")
normCal := CalibrateNormalization(corrected, d65SPD, "D65",
    100*time.Millisecond, Gain16x)
```

#### ApplyNormalization

Applies normalization to convert to physical units or reference illuminant.

```go
func (d *Device) ApplyNormalization(
    corrected CorrectedMeasurement,
    normCal NormalizationCalibration,
    currentIntegrationTime time.Duration, // Current measurement integration time
    currentGain Gain,                     // Current measurement gain
) (NormalizedMeasurement, error)

// Usage:
normalized := d.ApplyNormalization(corrected, calData.Normalization,
    raw.IntegrationTime, raw.Gain)
```

### 6. Spectral Reconstruction

#### CalibrateReconstructionMatrix

Calibrates matrix-based spectral reconstruction using reference spectrometer and test targets.

```go
func CalibrateReconstructionMatrix(
    variant Variant,
    sensorReadings []CorrectedMeasurement, // Sensor readings of test targets
    referenceSpectra []colorscience.SPD,   // Reference spectrometer spectra of same targets
    targetWavelengths vec.Vector,          // Target wavelengths for reconstruction
    method string,                         // "pseudo_inverse", "damped_least_squares", "wiener"
    lambda float32,                        // Regularization parameter (for damped LS)
) (ReconstructionMatrix, error)

// Requirements:
// - Number of test targets >= number of channels (for stable matrix)
// - Test targets should be linearly independent (diverse colors)
// - Reference spectrometer must cover target wavelength range

// Usage:
var sensorReadings []CorrectedMeasurement
var referenceSpectra []colorscience.SPD

// Measure multiple test targets (e.g., RGBWCMY patches)
for _, target := range testTargets {
    sensorReading := measureTargetWithAS734x(target)
    referenceSpectrum := measureTargetWithSpectrometer(target)
    sensorReadings = append(sensorReadings, sensorReading)
    referenceSpectra = append(referenceSpectra, referenceSpectrum)
}

wavelengths := vec.NewFrom(350.0, 351.0, ..., 1000.0)
reconMatrix := CalibrateReconstructionMatrix(VariantAS7341, sensorReadings,
    referenceSpectra, wavelengths, "damped_least_squares", 0.01)
```

#### ReconstructSpectrum

Reconstructs full spectrum from channel readings using calibrated matrix.

```go
func (d *Device) ReconstructSpectrum(
    normalized NormalizedMeasurement,
    reconMatrix ReconstructionMatrix,
) (colorscience.SPD, error)

// Usage:
spd, err := d.ReconstructSpectrum(normalized, *calData.ReconstructionMatrix)
```

#### ReconstructSpectrumFromChannels

Convenience method that uses filter response specs and reconstruction method.

```go
func (d *Device) ReconstructSpectrumFromChannels(
    corrected CorrectedMeasurement,
    filterCal FilterResponseCalibration,
    targetWavelengths vec.Vector,
    useDampedLS bool,
    lambda float32,
) (colorscience.SPD, error)

// Usage:
// Uses channel center/FWHM specs for reconstruction
wavelengths := vec.NewFrom(350.0, 351.0, ..., 1000.0)
spd, err := d.ReconstructSpectrumFromChannels(corrected, calData.FilterResponse,
    wavelengths, true, 0.01)
```

## Complete Calibration Workflow

### Step 1: Measure Dark Current

```go
darkCal := d.MeasureDarkCurrent(100, 100*time.Millisecond, Gain16x, 25.0)
calData.DarkCurrent = darkCal
```

### Step 2: Define Filter Responses

```go
// Option A: From center/FWHM specs (recommended for AS734x)
channels := getAS7341ChannelSpecs() // Helper function with datasheet specs
wavelengths := createWavelengthVector(350, 1000, 1) // 350-1000nm, 1nm steps
filterCal := LoadFilterResponseFromSpecs(VariantAS7341, channels,
    colorscience.FilterShapeSuperGaussian4, wavelengths)
calData.FilterResponse = filterCal

// Option B: From reference spectrometer measurements (most accurate)
// Requires measuring each channel's response with monochromatic light
```

### Step 3: Diffuser Compensation (if applicable)

```go
if hasDiffuser {
    diffuserSPD := loadDiffuserTransmission("diffuser.csv")
    calData.Diffuser = LoadDiffuserCalibration(diffuserSPD, "achromatic")
}
```

### Step 4: NIR Correction

```go
nirCal := d.CalibrateNIRCorrection(/* ... */)
calData.NIRCorrection = nirCal
```

### Step 5: Normalization

```go
d65SPD := colorscience.LoadIlluminantSPD("D65")
normCal := CalibrateNormalization(correctedMeasurement, d65SPD, "D65",
    100*time.Millisecond, Gain16x)
calData.Normalization = normCal
```

### Step 6: Matrix-Based Reconstruction (optional but recommended)

```go
// Measure test targets with both sensor and reference spectrometer
testTargets := []string{"R", "G", "B", "W", "C", "M", "Y"} // At least 8 for AS7341
var sensorReadings []CorrectedMeasurement
var referenceSpectra []colorscience.SPD

for _, target := range testTargets {
    // Measure with AS734x
    raw := d.Read()
    corrected := applyCorrections(raw, calData)
    sensorReadings = append(sensorReadings, corrected)
    
    // Measure with reference spectrometer
    refSpectrum := referenceSpectrometer.Measure(target)
    referenceSpectra = append(referenceSpectra, refSpectrum)
}

wavelengths := createWavelengthVector(350, 1000, 1)
reconMatrix := CalibrateReconstructionMatrix(VariantAS7341, sensorReadings,
    referenceSpectra, wavelengths, "damped_least_squares", 0.01)
calData.ReconstructionMatrix = reconMatrix
```

## Measurement Pipeline

### CorrectedMeasurement

Intermediate data structure after dark current and filter corrections.

```go
type CorrectedMeasurement struct {
    Timestamp        time.Time
    Variant          Variant
    Channels         []float32 // Corrected channel values
    IntegrationTime  time.Duration
    Gain             Gain
    Temperature      float32
}
```

### NormalizedMeasurement

After normalization to physical units.

```go
type NormalizedMeasurement struct {
    Timestamp        time.Time
    Variant          Variant
    Channels         []float32 // Normalized channel values (e.g., spectral irradiance)
    IntegrationTime  time.Duration
    Gain             Gain
    Illuminant       string // Reference illuminant used
}
```

### Complete Processing Chain

```go
func (d *Device) ProcessMeasurement(
    raw RawMeasurement,
    calData CalibrationData,
) (colorscience.SPD, error) {
    // Step 1: Dark current correction
    corrected, err := d.ApplyDarkCurrentCorrection(raw, calData.DarkCurrent, 25.0)
    if err != nil {
        return colorscience.SPD{}, err
    }
    
    // Step 2: Filter correction
    corrected, err = d.ApplyFilterCorrection(corrected, calData.FilterResponse)
    if err != nil {
        return colorscience.SPD{}, err
    }
    
    // Step 3: Diffuser compensation (if applicable)
    if calData.Diffuser != nil {
        corrected, err = d.ApplyDiffuserCompensation(corrected, *calData.Diffuser)
        if err != nil {
            return colorscience.SPD{}, err
        }
    }
    
    // Step 4: NIR correction
    if calData.NIRCorrection.Enabled {
        corrected, err = d.ApplyNIRCorrection(corrected, calData.NIRCorrection)
        if err != nil {
            return colorscience.SPD{}, err
        }
    }
    
    // Step 5: Normalization
    normalized, err := d.ApplyNormalization(corrected, calData.Normalization,
        raw.IntegrationTime, raw.Gain)
    if err != nil {
        return colorscience.SPD{}, err
    }
    
    // Step 6: Spectral reconstruction
    if calData.ReconstructionMatrix != nil {
        // Use matrix-based reconstruction (most accurate)
        spd, err := d.ReconstructSpectrum(normalized, *calData.ReconstructionMatrix)
        if err != nil {
            return colorscience.SPD{}, err
        }
        return spd, nil
    } else {
        // Use channel specs-based reconstruction
        wavelengths := createWavelengthVector(350, 1000, 1)
        spd, err := d.ReconstructSpectrumFromChannels(corrected, calData.FilterResponse,
            wavelengths, true, 0.01)
        if err != nil {
            return colorscience.SPD{}, err
        }
        return spd, nil
    }
}
```

## Calibration Persistence

### SaveCalibration

Saves calibration data to file (JSON or binary format).

```go
func SaveCalibration(calData CalibrationData, filename string) error

// Usage:
err := SaveCalibration(calData, "as7341_calibration.json")
```

### LoadCalibration

Loads calibration data from file.

```go
func LoadCalibration(filename string) (CalibrationData, error)

// Usage:
calData, err := LoadCalibration("as7341_calibration.json")
```

## Channel Specifications

### Helper Function for AS7341 Channels

```go
func GetAS7341ChannelSpecs() []colorscience.SensorChannel {
    return []colorscience.SensorChannel{
        {Name: "F1", CenterWL: 415, FWHM: 20},
        {Name: "F2", CenterWL: 445, FWHM: 20},
        {Name: "F3", CenterWL: 480, FWHM: 20},
        {Name: "F4", CenterWL: 515, FWHM: 20},
        {Name: "F5", CenterWL: 555, FWHM: 20},
        {Name: "F6", CenterWL: 590, FWHM: 20},
        {Name: "F7", CenterWL: 630, FWHM: 20},
        {Name: "F8", CenterWL: 680, FWHM: 20},
        {Name: "Clear", CenterWL: 0, FWHM: 0}, // No filter
        {Name: "NIR", CenterWL: 910, FWHM: 20},
    }
}
```

### Helper Function for AS7343 Channels

```go
func GetAS7343ChannelSpecs() []colorscience.SensorChannel {
    // AS7343 has 18 channels - define all center wavelengths and FWHM
    // (Specifications from datasheet)
}
```

## Best Practices

1. **Dark Current**: Measure periodically, especially after temperature changes
2. **Filter Response**: Use Super-Gaussian (n=4) shape for AS734x interference filters
3. **Matrix Reconstruction**: Use at least as many test targets as channels (preferably more)
4. **Linearly Independent Targets**: Use diverse colors (RGBWCMY) for stable matrix
5. **Reference Spectrometer**: Should be 10× more accurate than required sensor accuracy
6. **Calibration Storage**: Save calibration data with device serial number and timestamp
7. **Validation**: Verify calibration accuracy with known test targets

## References

- [ams Application Note AN000633: Spectral Sensor Calibration Methods](https://look.ams-osram.com/m/269928fe0dba7511/original/Spectral-Sensor-Calibration-Methods.pdf)
- [AS7341 Datasheet](https://look.ams-osram.com/m/24266a3e584de4db/original/AS7341-DS000504.pdf)
- ColorScience package: `github.com/itohio/EasyRobot/x/math/colorscience`
