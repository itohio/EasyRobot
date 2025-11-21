package render

import (
	"github.com/itohio/EasyRobot/x/math/colorscience"
)

// Patch represents a color patch for comparison
type Patch struct {
	Name         string
	Reference    colorscience.LAB // Reference LAB values
	Measured     colorscience.LAB // Measured LAB values
	ReferenceXYZ colorscience.XYZ // Reference XYZ values
	MeasuredXYZ  colorscience.XYZ // Measured XYZ values
	DeltaE       float32          // Delta E (ΔE) value
	RGB          colorscience.RGB // RGB color (from measured XYZ)
}

// CalculateDeltaE calculates and updates the Delta E value for the patch
func (p *Patch) CalculateDeltaE() {
	p.DeltaE = colorscience.DeltaE76(p.Reference, p.Measured)
}
