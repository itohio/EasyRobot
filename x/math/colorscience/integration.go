package colorscience

import (
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// IntegrationKernel computes the trapezoidal integration kernel for a given wavelength vector.
// The kernel can be used to compute ∫f(λ)·dλ ≈ kernel · f(λ) using dot product.
// Returns a vector where kernel[i] represents the weight for f(wavelengths[i]) in the integration.
func IntegrationKernel(wavelengths vecTypes.Vector) vec.Vector {
	wl := wavelengths.View().(vec.Vector)
	if wl.Len() == 0 {
		return vec.New(0)
	}
	if wl.Len() == 1 {
		// Single point - no integration possible
		return vec.New(1)
	}
	
	kernel := vec.New(wl.Len())
	
	// First point: half of first interval
	if wl.Len() > 1 {
		kernel[0] = (wl[1] - wl[0]) / 2.0
	}
	
	// Middle points: average of adjacent intervals
	for i := 1; i < wl.Len()-1; i++ {
		dLambda1 := wl[i] - wl[i-1]
		dLambda2 := wl[i+1] - wl[i]
		kernel[i] = (dLambda1 + dLambda2) / 2.0
	}
	
	// Last point: half of last interval
	if wl.Len() > 1 {
		kernel[wl.Len()-1] = (wl[wl.Len()-1] - wl[wl.Len()-2]) / 2.0
	}
	
	return kernel
}

// Integrate computes the integral of f(λ) over the wavelength range using trapezoidal rule.
// This is equivalent to kernel · f(λ) where kernel = IntegrationKernel(wavelengths).
func Integrate(wavelengths vecTypes.Vector, values vecTypes.Vector) float32 {
	kernel := IntegrationKernel(wavelengths)
	vals := values.View().(vec.Vector)
	
	if kernel.Len() != vals.Len() {
		panic("wavelengths and values must have the same length")
	}
	
	// Compute dot product: kernel · values
	return kernel.Dot(vals)
}

