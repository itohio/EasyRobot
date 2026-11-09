package dh

import (
	"fmt"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
)

///
/// FK and IK using Denavit-Hartenberg parameter table for homogenious transform matrix calculation.
/// How to construct DH parameter table:
/// https://www.youtube.com/watch?v=D3w3ZANOy3s (one prizmatic joint)
/// https://www.youtube.com/watch?v=dASdcqgBlqw (only rotational joints)
/// https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
///
/// DH table format:
///
///  n | theta | alpha | r | d | parameter index |
///  1 |
///
/// n - frame index
/// theta - rotation around Z previous axis
/// alpha - rotation around X previous axis
/// r - displacement along X axis
/// d - displacement along Z axis
/// parameter index - index of the column that joint parameter relates to:
///   0 = theta (revolute), 1 = alpha, 2 = r, 3 = d (prismatic)

const twoPi = 2 * math32.Pi

// Config is one row of a standard (distal) DH table:
// A = Rz(Theta)·Tz(D)·Tx(R)·Rx(Alpha), with the joint parameter added to the
// column selected by Index. Angles are radians, lengths metres; Min/Max bound
// the joint parameter in its own unit and must satisfy Min < Max (a zero-value
// Config would pin the joint at 0, so Validate rejects it).
type Config struct {
	Min   float32
	Max   float32
	Theta float32
	Alpha float32
	R     float32
	D     float32
	Index int
}

// Validate reports malformed configurations: non-finite fields, Min >= Max
// (including the zero value) and unsupported Index values. Errors wrap
// kintypes.ErrInvalidConfig.
func (c Config) Validate() error {
	for _, v := range [...]float32{c.Min, c.Max, c.Theta, c.Alpha, c.R, c.D} {
		if math32.IsNaN(v) || math32.IsInf(v, 0) {
			return fmt.Errorf("%w: dh: non-finite parameter in %+v", kintypes.ErrInvalidConfig, c)
		}
	}
	if !(c.Min < c.Max) {
		return fmt.Errorf("%w: dh: Min (%g) must be < Max (%g); set joint limits explicitly", kintypes.ErrInvalidConfig, c.Min, c.Max)
	}
	if c.Index < 0 || c.Index > 3 {
		return fmt.Errorf("%w: dh: Index %d not in {0 θ, 1 α, 2 r, 3 d}", kintypes.ErrInvalidConfig, c.Index)
	}
	return nil
}

func (c Config) Limit(a float32) float32 {
	switch {
	case a < c.Min:
		return c.Min
	case a > c.Max:
		return c.Max
	default:
		return a
	}
}

// angular reports whether the joint parameter is an angle (θ or α).
func (c Config) angular() bool {
	return c.Index == 0 || c.Index == 1
}

// continuous reports an angular joint whose limits cover a full turn, so it
// can never be blocked by a limit (it wraps instead).
func (c Config) continuous() bool {
	return c.angular() && c.Max-c.Min >= twoPi*(1-1e-6)
}

// wrapLimit maps an angular parameter into [Min, Max] modulo 2π when an
// equivalent angle lies inside the limits, otherwise clamps to the limit that
// is nearer on the circle. Linear parameters are clamped.
func (c Config) wrapLimit(a float32) float32 {
	if !c.angular() || (a >= c.Min && a <= c.Max) {
		return c.Limit(a)
	}
	off := a - c.Min
	off -= twoPi * math32.Floor(off/twoPi) // [0, 2π)
	w := c.Min + off
	if w <= c.Max {
		return w
	}
	if w-c.Max <= c.Min+twoPi-w {
		return c.Max
	}
	return c.Min
}

// maxOffset bounds the length of this joint's translation |(R, D)| over its
// limits; the sum over a chain bounds the distance from the base origin to the
// end effector.
func (c Config) maxOffset() float32 {
	r, d := math32.Abs(c.R), math32.Abs(c.D)
	switch c.Index {
	case 2:
		r = math32.Max(math32.Abs(c.R+c.Min), math32.Abs(c.R+c.Max))
	case 3:
		d = math32.Max(math32.Abs(c.D+c.Min), math32.Abs(c.D+c.Max))
	}
	return math32.Sqrt(r*r + d*d)
}

// CalculateTransform writes the joint's DH transform A for the parameter
// clamped to [Min, Max].
func (c Config) CalculateTransform(parameter float32, m *mat.Matrix4x4) bool {
	return c.transform(c.Limit(parameter), m)
}

// transform writes A without clamping the parameter.
func (c Config) transform(parameter float32, m *mat.Matrix4x4) bool {
	alpha := c.Alpha
	theta := c.Theta
	r := c.R
	d := c.D
	switch c.Index {
	case 0:
		theta += parameter
	case 1:
		alpha += parameter
	case 2:
		r += parameter
	case 3:
		d += parameter
	default:
		return false
	}
	ct := math32.Cos(theta)
	st := math32.Sin(theta)
	ca := math32.Cos(alpha)
	sa := math32.Sin(alpha)

	m[0][0] = ct
	m[0][1] = -st * ca
	m[0][2] = st * sa
	m[0][3] = r * ct
	m[1][0] = st
	m[1][1] = ct * ca
	m[1][2] = -ct * sa
	m[1][3] = r * st
	m[2][0] = 0
	m[2][1] = sa
	m[2][2] = ca
	m[2][3] = d
	m[3][0] = 0
	m[3][1] = 0
	m[3][2] = 0
	m[3][3] = 1
	return true
}
