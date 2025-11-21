package colorscience

import (
	"github.com/itohio/EasyRobot/x/math/vec"
)

// XYZ represents CIE XYZ tristimulus values.
type XYZ vec.Vector3D

// LAB represents CIE LAB color values.
type LAB vec.Vector3D

// RGB represents sRGB color values.
type RGB vec.Vector3D

// NewXYZ creates a new XYZ from X, Y, Z values.
func NewXYZ(X, Y, Z float32) XYZ {
	return XYZ{X, Y, Z}
}

// NewLAB creates a new LAB from L, a, b values.
func NewLAB(L, a, b float32) LAB {
	return LAB{L, a, b}
}

// NewRGB creates a new RGB from r, g, b values.
func NewRGB(r, g, b float32) RGB {
	return RGB{r, g, b}
}

// L returns the L component (lightness).
func (lab LAB) L() float32 {
	return float32(lab[0])
}

// A returns the a component (green-red axis).
func (lab LAB) A() float32 {
	return float32(lab[1])
}

// B returns the b component (blue-yellow axis).
func (lab LAB) B() float32 {
	return float32(lab[2])
}

// LAB returns the L, a, b components.
func (lab LAB) LAB() (float32, float32, float32) {
	return float32(lab[0]), float32(lab[1]), float32(lab[2])
}

// R returns the R component.
func (rgb RGB) R() float32 {
	return float32(rgb[0])
}

// G returns the G component.
func (rgb RGB) G() float32 {
	return float32(rgb[1])
}

// B returns the B component.
func (rgb RGB) B() float32 {
	return float32(rgb[2])
}

// RGB returns the r, g, b components.
func (rgb RGB) RGB() (float32, float32, float32) {
	return float32(rgb[0]), float32(rgb[1]), float32(rgb[2])
}

// ToLAB converts XYZ to LAB using the provided white point.
func (xyz XYZ) ToLAB(illuminant WhitePoint) LAB {
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	L, a, b := XYZToLAB(X, Y, Z, illuminant)
	return NewLAB(L, a, b)
}

// ToRGB converts XYZ to RGB.
// out255: if true, return values 0-255; if false, return 0-1.
func (xyz XYZ) ToRGB(out255 bool) RGB {
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	r, g, b := XYZToRGB(X, Y, Z, out255)
	return NewRGB(r, g, b)
}

// ToXYZ converts LAB to XYZ using the provided white point.
func (lab LAB) ToXYZ(illuminant WhitePoint) XYZ {
	L, a, b := lab.LAB()
	X, Y, Z := LABToXYZ(L, a, b, illuminant)
	return NewXYZ(X, Y, Z)
}

// ToRGB converts LAB to RGB using the provided white point.
// out255: if true, return values 0-255; if false, return 0-1.
func (lab LAB) ToRGB(illuminant WhitePoint, out255 bool) RGB {
	return lab.ToXYZ(illuminant).ToRGB(out255)
}

// ToXYZ converts RGB to XYZ.
func (rgb RGB) ToXYZ() XYZ {
	r, g, b := rgb.RGB()
	X, Y, Z := RGBToXYZ(r, g, b)
	return NewXYZ(X, Y, Z)
}

// ToLAB converts RGB to LAB using the provided white point.
func (rgb RGB) ToLAB(illuminant WhitePoint) LAB {
	return rgb.ToXYZ().ToLAB(illuminant)
}

// Luminance returns the luminance (Y component) of the XYZ value.
func (xyz XYZ) Luminance() float32 {
	return xyz[1]
}

// Adapt adapts XYZ values from source white point to destination white point.
// Returns adapted XYZ using the specified adaptation method.
func (xyz XYZ) Adapt(Ws, Wd WhitePoint, method AdaptationMethod) (XYZ, error) {
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	Xa, Ya, Za, err := AdaptXYZ(X, Y, Z, Ws, Wd, method)
	if err != nil {
		return XYZ{}, err
	}
	return NewXYZ(Xa, Ya, Za), nil
}

// DeltaE76 calculates the CIE76 color difference (ΔE) between two LAB color values.
// Formula: ΔE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
// Uses Vector3D.Distance() for efficient Euclidean distance calculation.
func DeltaE76(lab1, lab2 LAB) float32 {
	return vec.Vector3D(lab1).Distance(vec.Vector3D(lab2))
}
