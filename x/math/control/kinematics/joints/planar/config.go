package planar

import (
	"fmt"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
)

const (
	twoPi = 2 * math32.Pi
	// angleTol accepts solver angles this far (rad) outside a limit as inside
	// (they are clamped), absorbing float32 rounding at the boundary.
	angleTol = 1e-5
)

// Config describes one joint: angle limits in radians (Min < Max required; the
// zero value would pin the joint at 0) and the length of the link it drives.
type Config struct {
	Min    float32
	Max    float32
	Length float32
}

// Validate rejects non-finite fields, Min >= Max (including the zero value)
// and negative lengths. Errors wrap kintypes.ErrInvalidConfig.
func (c Config) Validate() error {
	for _, v := range [...]float32{c.Min, c.Max, c.Length} {
		if math32.IsNaN(v) || math32.IsInf(v, 0) {
			return fmt.Errorf("%w: planar: non-finite field in %+v", kintypes.ErrInvalidConfig, c)
		}
	}
	if !(c.Min < c.Max) {
		return fmt.Errorf("%w: planar: Min (%g) must be < Max (%g); set joint limits explicitly", kintypes.ErrInvalidConfig, c.Min, c.Max)
	}
	if c.Length < 0 {
		return fmt.Errorf("%w: planar: negative Length %g", kintypes.ErrInvalidConfig, c.Length)
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

// wrap maps angle a to its equivalent (mod 2π) inside [Min, Max] when one
// exists, with violation 0. Otherwise it returns the limit nearer on the
// circle and the angular distance to it.
func (c Config) wrap(a float32) (angle, violation float32) {
	if a >= c.Min-angleTol && a <= c.Max+angleTol {
		return c.Limit(a), 0
	}
	off := a - c.Min
	off -= twoPi * math32.Floor(off/twoPi) // [0, 2π)
	w := c.Min + off
	if w <= c.Max+angleTol {
		return c.Limit(w), 0
	}
	dMax, dMin := w-c.Max, c.Min+twoPi-w
	if dMax <= dMin {
		return c.Max, dMax
	}
	return c.Min, dMin
}
