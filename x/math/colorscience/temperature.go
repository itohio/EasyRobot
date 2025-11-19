package colorscience

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// ComputeColorTemperature computes the Correlated Color Temperature (CCT) and tint (duv) from an SPD.
// Returns CCT in Kelvin and duv (deviation from Planckian locus).
// Negative CCT indicates the chromaticity is outside the valid range.
//
// Algorithm: Uses McCamy's formula for CCT and calculates duv as distance from Planckian locus.
func (cs *ColorScience) ComputeColorTemperature(spd SPD) (cct float32, duv float32, err error) {
	if spd.Matrix == nil {
		return 0, 0, fmt.Errorf("SPD matrix cannot be nil")
	}

	// Compute XYZ from SPD
	xyz, err := cs.ComputeXYZ(spd.Matrix)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to compute XYZ: %w", err)
	}

	// Calculate chromaticity coordinates (x, y)
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	sum := X + Y + Z
	if sum == 0 {
		return 0, 0, fmt.Errorf("XYZ sum is zero")
	}

	x := float32(X) / sum
	y := float32(Y) / sum

	// Convert to CIE 1960 u,v coordinates for duv calculation
	// u = 4x / (-2x + 12y + 3)
	// v = 6y / (-2x + 12y + 3)
	denom := -2*x + 12*y + 3
	if denom == 0 {
		return 0, 0, fmt.Errorf("invalid chromaticity coordinates")
	}
	u := 4 * x / denom
	v := 6 * y / denom

	// Calculate CCT using McCamy's formula
	// CCT = 449*n^3 + 3525*n^2 + 6823.3*n + 5520.33
	// where n = (x - 0.3320) / (0.1858 - y)
	denomCCT := 0.1858 - y
	if denomCCT == 0 {
		return 0, 0, fmt.Errorf("invalid chromaticity for CCT calculation")
	}
	n := (x - 0.3320) / denomCCT
	cct = 449*n*n*n + 3525*n*n + 6823.3*n + 5520.33

	// For temperatures below ~3000K or above ~7000K, McCamy's formula may be inaccurate
	// Use alternative: Robertson's method for better accuracy
	if cct < 3000 || cct > 7000 {
		cct, err = computeCCTRobertson(x, y)
		if err != nil {
			return 0, 0, err
		}
	}

	// Calculate duv (distance from Planckian locus in u,v space)
	// Find the closest point on the Planckian locus at the computed CCT
	planckU, planckV, err := planckianUV(cct)
	if err != nil {
		return cct, 0, err // Return CCT even if duv calculation fails
	}

	// duv is the signed distance perpendicular to the Planckian locus
	duv = float32(math.Sqrt(float64((u-planckU)*(u-planckU) + (v-planckV)*(v-planckV))))

	// Determine sign: positive if above locus (greenish), negative if below (pinkish)
	// Approximate by checking if the point is above the line
	// For simplicity, use the angle from the Planckian locus
	thetaPlanck := math.Atan2(float64(planckV), float64(planckU))
	thetaPoint := math.Atan2(float64(v), float64(u))
	diff := thetaPoint - thetaPlanck
	if diff < -math.Pi {
		diff += 2 * math.Pi
	}
	if diff > math.Pi {
		diff -= 2 * math.Pi
	}
	if diff < 0 {
		duv = -duv
	}

	return cct, duv, nil
}

// computeCCTRobertson uses Robertson's method for CCT calculation.
// This is more accurate for extreme temperatures but requires iterative search.
func computeCCTRobertson(x, y float32) (float32, error) {
	// Robertson's method: find the temperature where the distance from (x,y) to the Planckian locus is minimized
	// Use bisection search between reasonable bounds
	minT := float32(1000)  // Minimum temperature (K)
	maxT := float32(20000) // Maximum temperature (K)

	// Find u,v for the given x,y
	denom := -2*x + 12*y + 3
	if denom == 0 {
		return 0, fmt.Errorf("invalid chromaticity coordinates")
	}
	u := 4 * x / denom
	v := 6 * y / denom

	// Binary search for the closest temperature
	bestT := float32(6500) // Default
	bestDist := float32(math.MaxFloat32)

	for iter := 0; iter < 50; iter++ {
		midT := (minT + maxT) / 2
		planckU, planckV, err := planckianUV(midT)
		if err != nil {
			return 0, err
		}

		dist := float32(math.Sqrt(float64((u-planckU)*(u-planckU) + (v-planckV)*(v-planckV))))
		if dist < bestDist {
			bestDist = dist
			bestT = midT
		}

		// Compare with one endpoint to decide which half to search
		t1 := midT
		t2 := midT + (maxT-minT)/4
		if t2 > maxT {
			t2 = maxT
		}

		planckU1, planckV1, _ := planckianUV(t1)
		planckU2, planckV2, _ := planckianUV(t2)
		dist1 := float32(math.Sqrt(float64((u-planckU1)*(u-planckU1) + (v-planckV1)*(v-planckV1))))
		dist2 := float32(math.Sqrt(float64((u-planckU2)*(u-planckU2) + (v-planckV2)*(v-planckV2))))

		if dist1 < dist2 {
			maxT = midT
		} else {
			minT = midT
		}

		if maxT-minT < 0.1 {
			break
		}
	}

	return bestT, nil
}

// planckianUV returns the u,v coordinates on the Planckian locus for a given temperature.
func planckianUV(t float32) (u, v float32, err error) {
	if t <= 0 {
		return 0, 0, fmt.Errorf("temperature must be positive")
	}

	// For Planckian radiator, calculate chromaticity from temperature
	// Use blackbody radiation formula
	// Approximate using McCamy's formula backwards or direct calculation

	// Direct calculation from temperature to x,y (Planckian locus)
	// x_D = -0.2661239e9/T^3 - 0.2343580e6/T^2 + 0.8776956e3/T + 0.179910
	// for T in [1667, 40000]
	T := float64(t)
	var x float64
	if T >= 1667 && T <= 40000 {
		x = -0.2661239e9/(T*T*T) - 0.2343580e6/(T*T) + 0.8776956e3/T + 0.179910
	} else {
		// Outside valid range, use approximation
		if T < 1667 {
			T = 1667
		} else {
			T = 40000
		}
		x = -0.2661239e9/(T*T*T) - 0.2343580e6/(T*T) + 0.8776956e3/T + 0.179910
	}

	// y_D = -1.1063814*x^3 - 1.34811020*x^2 + 2.18555832*x - 0.20219683
	// for x in [0.1858, 0.5781]
	xF := float32(x)
	var yF float32
	if xF >= 0.1858 && xF <= 0.5781 {
		yF = -1.1063814*xF*xF*xF - 1.34811020*xF*xF + 2.18555832*xF - 0.20219683
	} else {
		// Outside valid range, use linear approximation
		if xF < 0.1858 {
			yF = 0.4425*xF + 0.1548
		} else {
			yF = 3.2944*xF - 1.6214
		}
	}

	// Convert x,y to u,v
	denom := -2*xF + 12*yF + 3
	if denom == 0 {
		return 0, 0, fmt.Errorf("invalid chromaticity from temperature")
	}
	u = 4 * xF / denom
	v = 6 * yF / denom

	return u, v, nil
}

// ComputeCRI computes the Color Rendering Index (CRI) of an SPD as a light source.
// Returns the general CRI (Ra, average of 8 test color samples).
// CRI ranges from 0 to 100, with higher values indicating better color rendering.
//
// The calculation compares the color rendering of test color samples under the test SPD
// versus under a reference illuminant with the same CCT.
//
// Parameters:
//   - spd: SPD to compute CRI for (as a light source)
//
// Returns: CRI Ra (0-100), error
func (cs *ColorScience) ComputeCRI(spd SPD) (float32, error) {
	if spd.Matrix == nil {
		return 0, fmt.Errorf("SPD matrix cannot be nil")
	}

	// First, compute the CCT of the test SPD
	cct, _, err := cs.ComputeColorTemperature(spd)
	if err != nil {
		return 0, fmt.Errorf("failed to compute CCT: %w", err)
	}

	if cct <= 0 || cct < 1000 || cct > 20000 {
		return 0, fmt.Errorf("invalid CCT: %f K (must be between 1000-20000K)", cct)
	}

	// Get reference illuminant based on CCT
	// For CCT < 5000K: use Planckian radiator (blackbody)
	// For CCT >= 5000K: use daylight illuminant
	var referenceSPD SPD
	if cct < 5000 {
		// Use Planckian radiator
		var err error
		referenceSPD, err = generatePlanckianSPD(cct, cs.cmf.WavelengthsValues())
		if err != nil {
			return 0, fmt.Errorf("failed to generate Planckian SPD: %w", err)
		}
	} else {
		// Use daylight illuminant closest to CCT
		// For simplicity, use D65 or interpolate between D50 and D65
		daylightSPD, err := LoadIlluminantSPD("D65")
		if err != nil {
			return 0, fmt.Errorf("failed to load daylight illuminant: %w", err)
		}
		// TODO: Better approach would be to use CIE D illuminant at exact CCT
		referenceSPD = daylightSPD
	}

	// Get the 8 CIE test color samples (TCS 1-8)
	tcsList := getCIETestColorSamples()
	if len(tcsList) < 8 {
		return 0, fmt.Errorf("insufficient test color samples")
	}

	// Calculate color differences for each test color sample
	var riSum float32
	validSamples := 0

	for i := 0; i < 8; i++ {
		tcs := tcsList[i]

		// Compute XYZ under test illuminant
		testXYZ, err := cs.computeXYZReflective(spd, tcs)
		if err != nil {
			continue // Skip this sample
		}

		// Compute XYZ under reference illuminant
		refXYZ, err := cs.computeXYZReflective(referenceSPD, tcs)
		if err != nil {
			continue // Skip this sample
		}

		// Apply chromatic adaptation (both should have same CCT, so minimal adaptation needed)
		// For simplicity, skip adaptation since reference should match test CCT

		// Convert to LAB color space
		whitePoint := cs.WhitePoint()
		testLAB := testXYZ.ToLAB(whitePoint)
		refLAB := refXYZ.ToLAB(whitePoint)

		// Calculate color difference (ΔE) using CIE76 formula
		// ΔE = sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
		L1, a1, b1 := vec.Vector3D(testLAB).XYZ()
		L2, a2, b2 := vec.Vector3D(refLAB).XYZ()
		deltaE := float32(math.Sqrt(float64((L1-L2)*(L1-L2) + (a1-a2)*(a1-a2) + (b1-b2)*(b1-b2))))

		// Calculate special CRI: Ri = 100 - 4.6 * ΔE
		Ri := 100 - 4.6*deltaE
		if Ri < 0 {
			Ri = 0
		}
		if Ri > 100 {
			Ri = 100
		}

		riSum += Ri
		validSamples++
	}

	if validSamples == 0 {
		return 0, fmt.Errorf("failed to compute CRI for any test color samples")
	}

	// General CRI (Ra) is the average of the 8 special CRIs
	ra := riSum / float32(validSamples)

	return ra, nil
}

// computeXYZReflective computes XYZ for a reflective sample under an illuminant.
func (cs *ColorScience) computeXYZReflective(illuminantSPD, reflectanceSPD SPD) (XYZ, error) {
	// The product SPD is: I(λ) = S(λ) * E(λ) where S is reflectance, E is illuminant
	// Create product SPD: reflectance * illuminant
	illumWl := illuminantSPD.Wavelengths()
	illumVals := illuminantSPD.Values()
	reflWl := reflectanceSPD.Wavelengths()
	reflVals := reflectanceSPD.Values()

	if illumWl == nil || illumVals == nil || reflWl == nil || reflVals == nil {
		return XYZ{}, fmt.Errorf("illuminant and reflectance SPDs must have wavelengths and values")
	}

	// Interpolate reflectance to illuminant wavelengths
	reflInterp := reflectanceSPD.Interpolate(vec.Vector(illumWl))
	reflInterpVals := reflInterp.Values()

	// Create product: I(λ) = S(λ) * E(λ)
	productVals := vec.New(illumWl.Len())
	for i := 0; i < illumWl.Len(); i++ {
		productVals[i] = reflInterpVals[i] * illumVals[i]
	}

	// Create product SPD matrix (1 row: values only, wavelengths match CMF/illuminant)
	productMat := mat.New(1, illumWl.Len())
	productMat.SetRow(0, productVals)

	return cs.ComputeXYZ(productMat)
}

// generatePlanckianSPD generates a Planckian (blackbody) SPD at the given temperature.
func generatePlanckianSPD(t float32, wavelengths vec.Vector) (SPD, error) {
	if wavelengths == nil || wavelengths.Len() == 0 {
		return SPD{}, fmt.Errorf("wavelengths cannot be nil or empty")
	}

	// Planck's law: B(λ,T) = (2*h*c^2/λ^5) * 1/(exp(h*c/(λ*k*T)) - 1)
	// Constants:
	// h = 6.62607015e-34 J*s (Planck constant)
	// c = 299792458 m/s (speed of light)
	// k = 1.380649e-23 J/K (Boltzmann constant)
	//
	// For spectral radiance in W/(m^2*sr*nm):
	// B(λ,T) = 1.191042972e-16 / (λ^5 * (exp(1.438776877e-2/(λ*T)) - 1))
	// where λ is in meters
	//
	// For normalized SPD (relative), we can simplify

	h := 6.62607015e-34
	c := 299792458.0
	k := 1.380649e-23

	values := vec.New(wavelengths.Len())
	T := float64(t)

	for i := 0; i < wavelengths.Len(); i++ {
		lambdaM := float64(wavelengths[i]) * 1e-9 // Convert nm to meters
		if lambdaM <= 0 {
			return SPD{}, fmt.Errorf("wavelength must be positive")
		}

		// Planck's law
		expTerm := math.Exp(h*c/(lambdaM*k*T)) - 1.0
		if expTerm == 0 {
			values[i] = 0
			continue
		}

		// Spectral radiance: B(λ,T) = (2*h*c^2/λ^5) / expTerm
		b := (2.0 * h * c * c) / (lambdaM * lambdaM * lambdaM * lambdaM * lambdaM * expTerm)

		values[i] = float32(b)
	}

	// Normalize to peak = 1.0 for relative SPD
	maxVal := float32(0)
	for i := 0; i < values.Len(); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
		}
	}
	if maxVal > 0 {
		for i := 0; i < values.Len(); i++ {
			values[i] /= maxVal
		}
	}

	return NewSPD(wavelengths, values), nil
}

// getCIETestColorSamples returns the 8 CIE test color sample reflectance spectra.
// These are simplified approximations. For accurate CRI, use full reflectance data.
// TODO: Load actual CIE test color sample reflectance data from embedded files
func getCIETestColorSamples() []SPD {
	// CIE TCS reflectance data would normally be loaded from data files
	// For now, return placeholder - this needs proper test color sample data
	// The 8 samples are typically:
	// TCS1: Light greyish red
	// TCS2: Dark greyish yellow
	// TCS3: Strong yellow green
	// TCS4: Moderate yellowish green
	// TCS5: Light bluish green
	// TCS6: Light blue
	// TCS7: Light violet
	// TCS8: Light reddish purple

	// Placeholder - actual implementation would load from data files
	return []SPD{}
}
