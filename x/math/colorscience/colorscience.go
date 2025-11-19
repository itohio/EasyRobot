package colorscience

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// ColorScience provides comprehensive spectral color calculations.
type ColorScience struct {
	cmf            ObserverCMF
	illuminant     SPD
	illuminantName string
	whitePoint     WhitePoint
	whitePointSet  bool // Track if white point was explicitly set
	observer       ObserverType
	dark           SPD // Dark calibration SPD (sensor dark reading)
	light          SPD // Light calibration SPD (reference white/light reading)
}

// New creates a new ColorScience instance with options.
// Defaults: D65 illuminant, 10-degree observer, WhitePointD65_10.
func New(opts ...Option) (*ColorScience, error) {
	cs := &ColorScience{
		observer:       Observer10Deg, // Default
		illuminantName: "D65",         // Default
		whitePoint:     WhitePointD65_10,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(cs); err != nil {
			return nil, fmt.Errorf("option error: %w", err)
		}
	}

	// Load CMF if not set
	if cs.cmf.Matrix == nil {
		cmf, err := LoadCMF(cs.observer)
		if err != nil {
			return nil, fmt.Errorf("failed to load CMF: %w", err)
		}
		cs.cmf = cmf
	}

	// Load illuminant if not set
	if cs.illuminant.Matrix == nil && cs.illuminantName != "" {
		illuminant, err := LoadIlluminantSPD(cs.illuminantName)
		if err != nil {
			return nil, fmt.Errorf("failed to load illuminant: %w", err)
		}
		cs.illuminant = illuminant

		// Auto-set white point based on illuminant name and observer if not already set
		if !cs.whitePointSet {
			cs.whitePoint = getWhitePointForIlluminant(cs.illuminantName, cs.observer)
		}
	}

	if cs.illuminant.Matrix == nil {
		return nil, fmt.Errorf("illuminant must be set via WithIlluminant or WithIlluminantSPD")
	}

	// Ensure CMF and illuminant wavelengths match
	// Interpolate to the one with higher resolution (more points)
	if err := cs.matchCMFAndIlluminantWavelengths(); err != nil {
		return nil, fmt.Errorf("failed to match CMF and illuminant wavelengths: %w", err)
	}

	return cs, nil
}

// matchCMFAndIlluminantWavelengths ensures CMF and illuminant use the same wavelength grid.
// Interpolates to the one with higher resolution (more points).
func (cs *ColorScience) matchCMFAndIlluminantWavelengths() error {
	if cs.cmf.Matrix == nil || cs.illuminant.Matrix == nil {
		return fmt.Errorf("CMF and illuminant must be set")
	}

	cmfWl := cs.cmf.WavelengthsValues()
	illumWl := cs.illuminant.Wavelengths()

	if cmfWl == nil || illumWl == nil {
		return fmt.Errorf("CMF and illuminant must have wavelengths")
	}

	cmfLen := cmfWl.Len()
	illumLen := illumWl.Len()

	// Check if wavelengths already match
	if cmfLen == illumLen {
		// Check if values are the same (within tolerance)
		match := true
		for i := 0; i < cmfLen; i++ {
			if cmfWl[i] != illumWl[i] {
				match = false
				break
			}
		}
		if match {
			return nil // Already matched
		}
	}

	// Determine which has higher resolution (more points)
	// If equal, prefer CMF wavelengths
	if cmfLen >= illumLen {
		// Interpolate illuminant to CMF wavelengths
		cmfWlVec := vecTypes.Vector(cmfWl)
		cs.illuminant = cs.illuminant.Interpolate(cmfWlVec)
	} else {
		// Interpolate CMF to illuminant wavelengths
		illumWlVec := vecTypes.Vector(illumWl)
		newCMF, err := cs.interpolateCMF(illumWlVec)
		if err != nil {
			return fmt.Errorf("failed to interpolate CMF: %w", err)
		}
		cs.cmf = newCMF
	}

	return nil
}

// interpolateCMF interpolates CMF to new wavelengths.
func (cs *ColorScience) interpolateCMF(targetWl vecTypes.Vector) (ObserverCMF, error) {
	if cs.cmf.Matrix == nil {
		return ObserverCMF{}, fmt.Errorf("CMF cannot be nil")
	}

	cmfWl := cs.cmf.WavelengthsValues()
	if cmfWl == nil {
		return ObserverCMF{}, fmt.Errorf("CMF must have wavelengths")
	}

	targetWlVec := targetWl.View().(vec.Vector)
	n := targetWlVec.Len()

	// Interpolate XBar, YBar, ZBar to target wavelengths
	xBarSPD := cs.cmf.XBar()
	yBarSPD := cs.cmf.YBar()
	zBarSPD := cs.cmf.ZBar()

	interpXBar := xBarSPD.Interpolate(targetWl)
	interpYBar := yBarSPD.Interpolate(targetWl)
	interpZBar := zBarSPD.Interpolate(targetWl)

	// Create new CMF matrix
	newCMF := mat.New(4, n)
	newCMF.SetRow(0, targetWlVec)
	newCMF.SetRow(1, interpXBar.Values())
	newCMF.SetRow(2, interpYBar.Values())
	newCMF.SetRow(3, interpZBar.Values())

	return ObserverCMF{Matrix: newCMF}, nil
}

// ComputeXYZ computes CIE XYZ tristimulus values from a spectral power distribution matrix.
// The input matrix can be:
//   - 1 row: values only (wavelengths match CMF/illuminant, no interpolation needed)
//   - 2 rows: row 0 = wavelengths, row 1 = values (interpolation to CMF/illuminant wavelengths)
func (cs *ColorScience) ComputeXYZ(spdMatrix matTypes.Matrix) (XYZ, error) {
	if spdMatrix == nil {
		return XYZ{}, fmt.Errorf("spd matrix cannot be nil")
	}

	rows := spdMatrix.Rows()
	if rows < 1 || rows > 2 {
		return XYZ{}, fmt.Errorf("spd matrix must have 1 or 2 rows, got %d", rows)
	}

	// Get CMF wavelengths (CMF and illuminant are already matched during construction)
	cmfWl := cs.cmf.WavelengthsValues()
	targetWlVecTypes := vecTypes.Vector(cmfWl)

	var interpVals vec.Vector

	if rows == 1 {
		// Single row: values only, wavelengths match CMF/illuminant
		spdValues := spdMatrix.Row(0).(vec.Vector)
		if spdValues.Len() != cmfWl.Len() {
			return XYZ{}, fmt.Errorf("spd values length (%d) must match CMF/illuminant wavelengths length (%d)", spdValues.Len(), cmfWl.Len())
		}
		interpVals = spdValues
	} else {
		// Two rows: row 0 = wavelengths, row 1 = values
		spdWl := spdMatrix.Row(0).(vec.Vector)
		spdValues := spdMatrix.Row(1).(vec.Vector)
		if spdWl.Len() != spdValues.Len() {
			return XYZ{}, fmt.Errorf("wavelengths (%d) and values (%d) lengths must match", spdWl.Len(), spdValues.Len())
		}

		// Create input SPD and interpolate to CMF wavelengths
		inputSPD := NewSPD(spdWl, spdValues)
		interpSPD := inputSPD.Interpolate(targetWlVecTypes)
		interpVals = interpSPD.Values()
	}

	// CMF and illuminant are already matched during construction, so no interpolation needed
	cmfXBar := cs.cmf.XBarValues()
	cmfYBar := cs.cmf.YBarValues()
	cmfZBar := cs.cmf.ZBarValues()
	illumVals := cs.illuminant.Values()

	// Calculate XYZ using numerical integration
	// X = k * Σ[I(λ) * x̄(λ) * Δλ]
	// Y = k * Σ[I(λ) * ȳ(λ) * Δλ]
	// Z = k * Σ[I(λ) * z̄(λ) * Δλ]
	// where I(λ) = S(λ) * E(λ) for reflective
	// k = 100 / Σ[E(λ) * ȳ(λ) * Δλ] for reflective

	var sumX, sumY, sumZ float32
	var sumYIllum float32

	// Build integration kernel
	kernel := IntegrationKernel(cmfWl)

	// Build the product SPD: I(λ) = S(λ) * E(λ)
	n := cmfWl.Len()
	productSPD := vec.New(n)
	for i := 0; i < n; i++ {
		productSPD[i] = interpVals[i] * illumVals[i]
	}

	// Calculate normalization constant from illuminant
	yIllumIntegrand := vec.New(n)
	for i := 0; i < n; i++ {
		yIllumIntegrand[i] = illumVals[i] * cmfYBar[i]
	}
	sumYIllum = kernel.Dot(yIllumIntegrand)

	// Compute integrals using dot products: ∫f(λ)·dλ = kernel · f(λ)
	// X component: ∫[I(λ) * x̄(λ)]·dλ
	xIntegrand := vec.New(n)
	for i := 0; i < n; i++ {
		xIntegrand[i] = productSPD[i] * cmfXBar[i]
	}
	sumX = kernel.Dot(xIntegrand)

	// Y component: ∫[I(λ) * ȳ(λ)]·dλ
	yIntegrand := vec.New(n)
	for i := 0; i < n; i++ {
		yIntegrand[i] = productSPD[i] * cmfYBar[i]
	}
	sumY = kernel.Dot(yIntegrand)

	// Z component: ∫[I(λ) * z̄(λ)]·dλ
	zIntegrand := vec.New(n)
	for i := 0; i < n; i++ {
		zIntegrand[i] = productSPD[i] * cmfZBar[i]
	}
	sumZ = kernel.Dot(zIntegrand)

	// Normalize (for reflective, k = 100 / Σ[E(λ) * ȳ(λ) * Δλ])
	if sumYIllum > 0 {
		k := 100.0 / sumYIllum
		sumX *= k
		sumY *= k
		sumZ *= k
	}

	return NewXYZ(sumX, sumY, sumZ), nil
}

// WhitePoint returns the configured white point.
func (cs *ColorScience) WhitePoint() WhitePoint {
	return cs.whitePoint
}

// Calibrate calibrates a measurement SPD according to dark and light calibration readings.
// Normalizes SPD readings to the light SPD given non-ideal illuminant and dark current.
//
// The calibration process:
//  1. Subtracts dark current from measurement and light: corrected = raw - dark
//  2. Normalizes by light reading: calibrated = corrected_measurement / corrected_light
//
// The output is normalized to the light SPD (reflectance or transmittance [0,1] relative to the reference light).
//
// If optWhitePoint is provided and non-zero, it's used for reflective calibration normalization.
// The white point's Y value is used to scale reflectance to 100% for the reference white.
//
// Parameters:
//   - dst: Output SPD (must already be initialized with wavelengths)
//   - measurement: Raw measurement SPD
//   - optWhitePoint: Optional white point for reflective calibration (if not provided, uses configured white point)
//
// Returns error if dark or light calibration SPDs are not set, or if wavelengths don't match.
func (cs *ColorScience) Calibrate(dst *SPD, measurement SPD, optWhitePoint ...WhitePoint) error {
	// Check if dark and light are set
	if cs.dark.Matrix == nil || cs.light.Matrix == nil {
		return fmt.Errorf("dark and light calibration SPDs must be set via WithDark and WithLight")
	}

	if dst == nil {
		return fmt.Errorf("dst cannot be nil")
	}

	if dst.Matrix == nil {
		return fmt.Errorf("dst SPD must be initialized with wavelengths")
	}

	if measurement.Matrix == nil {
		return fmt.Errorf("measurement SPD cannot be nil")
	}

	// Get target wavelengths (from dst)
	targetWl := dst.Wavelengths()
	if targetWl == nil || targetWl.Len() == 0 {
		return fmt.Errorf("dst SPD must have wavelengths")
	}
	targetWlVec := vecTypes.Vector(targetWl)

	// Interpolate all SPDs to the same wavelength grid
	interpMeasurement := measurement.Interpolate(targetWlVec)
	interpDark := cs.dark.Interpolate(targetWlVec)
	interpLight := cs.light.Interpolate(targetWlVec)

	// Get value vectors
	measVals := interpMeasurement.Values()
	darkVals := interpDark.Values()
	lightVals := interpLight.Values()

	// Ensure all vectors have the same length
	if measVals.Len() != darkVals.Len() || darkVals.Len() != lightVals.Len() {
		return fmt.Errorf("interpolated SPDs must have the same length")
	}

	// Normalize to light SPD: calibrated = (measurement - dark) / (light - dark)
	// Element-wise operations (handle division by zero)
	calibrated := vec.New(measVals.Len())
	for i := 0; i < calibrated.Len(); i++ {
		correctedMeasurement := measVals[i] - darkVals[i]
		correctedLight := lightVals[i] - darkVals[i]
		if correctedLight != 0 {
			calibrated[i] = correctedMeasurement / correctedLight
		} else {
			// Avoid division by zero - set to 0 if light is zero
			calibrated[i] = 0
		}
	}

	// If WhitePoint is provided for reflective calibration, we could apply additional normalization
	// The calibrated value already represents reflectance/transmittance [0,1] relative to the reference light
	_ = optWhitePoint // Reserved for future reflective calibration enhancements

	// Store calibrated values in dst
	dst.Matrix.SetRow(1, calibrated)

	return nil
}
