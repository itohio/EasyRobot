package colorscience

import (
	"math"

	"github.com/chewxy/math32"
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

// DeltaE94 calculates the CIE94 color difference (ΔE94) between two LAB color values.
// Reference: https://en.wikipedia.org/wiki/Color_difference#CIE94
// ApplicationType: "graphic" or "textiles" (graphic: K_L=1, K1=0.045, K2=0.015; textiles: K_L=2, K1=0.048, K2=0.014)
func DeltaE94(lab1, lab2 LAB, applicationType string) float32 {
	var (
		K_L, K1, K2 float32
	)
	switch applicationType {
	case "textiles":
		K_L, K1, K2 = 2, 0.048, 0.014
	default: // "graphic", default
		K_L, K1, K2 = 1, 0.045, 0.015
	}
	L1, a1, b1 := lab1.LAB()
	L2, a2, b2 := lab2.LAB()
	dL := L1 - L2
	C1 := float32(hypot(float64(a1), float64(b1)))
	C2 := float32(hypot(float64(a2), float64(b2)))
	dC := C1 - C2
	da := a1 - a2
	db := b1 - b2
	dH2 := da*da + db*db - dC*dC
	if dH2 < 0 {
		dH2 = 0
	}
	Vc := 1 + K1*C1
	Vh := 1 + K2*C1

	return float32(
		hypot3(
			float64(dL)/float64(K_L),
			float64(dC)/float64(Vc),
			float64(float32Sqrt(dH2))/float64(Vh),
		),
	)
}

// DeltaE2000 calculates the CIEDE2000 color difference (ΔE00) between two LAB color values.
// Reference: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
func DeltaE2000(lab1, lab2 LAB) float32 {
	L1, a1, b1 := lab1.LAB()
	L2, a2, b2 := lab2.LAB()

	C1 := float32(hypot(float64(a1), float64(b1)))
	C2 := float32(hypot(float64(a2), float64(b2)))
	CBar := (C1 + C2) / 2
	G := float32(0.5 * (1 - float64(float32Sqrt(CBar/float32(7*CBar*CBar+25*25)))))
	a1p := a1 * (1 + G)
	a2p := a2 * (1 + G)
	C1p := float32(hypot(float64(a1p), float64(b1)))
	C2p := float32(hypot(float64(a2p), float64(b2)))
	CBarp := (C1p + C2p) / 2

	h1p := atan2Deg(b1, a1p)
	h2p := atan2Deg(b2, a2p)

	dLp := L2 - L1
	dCp := C2p - C1p

	var dhp float32
	if C1p*C2p == 0 {
		dhp = 0
	} else {
		dh := h2p - h1p
		switch {
		case dh > 180:
			dh -= 360
		case dh < -180:
			dh += 360
		}
		dhp = dh
	}
	dHp := 2 * float32Sqrt(float32(C1p)*float32(C2p)) * float32SinDeg(dhp/2)

	LBarp := (L1 + L2) / 2
	CBarp = (C1p + C2p) / 2

	var hBarp float32
	if C1p*C2p == 0 {
		hBarp = h1p + h2p
	} else {
		dh := h1p - h2p
		sum := h1p + h2p
		switch {
		case float32Abs(dh) > 180:
			if sum < 360 {
				hBarp = (sum + 360) / 2
			} else {
				hBarp = (sum - 360) / 2
			}
		default:
			hBarp = sum / 2
		}
	}

	T := 1 - 0.17*float32CosDeg(hBarp-30) +
		0.24*float32CosDeg(2*hBarp) +
		0.32*float32CosDeg(3*hBarp+6) -
		0.20*float32CosDeg(4*hBarp-63)

	dTheta := 30.0 * float32Exp(-((hBarp-275)/25)*((hBarp-275)/25))
	Rc := 2 * float32Sqrt(CBarp*CBarp*CBarp*CBarp*CBarp*CBarp*CBarp) / (CBarp*CBarp + 25*25)
	Sl := 1 + ((0.015 * (LBarp - 50) * (LBarp - 50)) / float32Sqrt(20+(LBarp-50)*(LBarp-50)))
	Sc := 1 + 0.045*CBarp
	Sh := 1 + 0.015*CBarp*T
	Rt := -float32SinDeg(2*dTheta) * Rc

	// Delta E 2000
	return float32Sqrt(
		(dLp/Sl)*(dLp/Sl) +
			(dCp/Sc)*(dCp/Sc) +
			(dHp/Sh)*(dHp/Sh) +
			Rt*(dCp/Sc)*(dHp/Sh),
	)
}

// --- Helpers ---

func hypot(x, y float64) float64 {
	return math.Sqrt(x*x + y*y)
}

func hypot3(x, y, z float64) float64 {
	return math.Sqrt(x*x + y*y + z*z)
}

func float32Sqrt(x float32) float32 {
	if x < 0 {
		return 0
	}
	return math32.Sqrt(x)
}

func float32Abs(x float32) float32 {
	return math32.Abs(x)
}

func float32SinDeg(deg float32) float32 {
	return math32.Sin(deg * (math.Pi / 180))
}

func float32CosDeg(deg float32) float32 {
	return math32.Cos(deg * (math.Pi / 180))
}

func float32Exp(x float32) float32 {
	return math32.Exp(x)
}

func atan2Deg(y, x float32) float32 {
	return math32.Atan2(y, x) * (180 / math.Pi)
}
