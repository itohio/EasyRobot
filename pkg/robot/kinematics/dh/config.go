package dh

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/mat"
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
/// parameter index - index of the column that joint parameter relates to (either 0 or 3 for rotational and prizmatic respectively)

type Config struct {
	Min   float32
	Max   float32
	Theta float32
	Alpha float32
	R     float32
	D     float32
	Index int
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

func (c Config) CalculateTransform(parameter float32, m *mat.Matrix4x4) bool {
	parameter = c.Limit(parameter)
	alpha := c.Alpha
	theta := c.Theta
	r := c.R
	d := c.D
	switch c.Index {
	case 0:
		theta += parameter
		break
	case 1:
		alpha += parameter
		break
	case 2:
		r += parameter
		break
	case 3:
		d += parameter
		break
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
