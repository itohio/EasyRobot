package planner

import (
	"errors"
	stdmath "math"

	ermath "github.com/itohio/EasyRobot/pkg/core/math"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

var (
	// ErrInvalidPath indicates the provided path is empty or degenerate.
	ErrInvalidPath = errors.New("motion/planner: invalid path")
	// ErrInvalidTrajectory indicates the desired trajectory is empty.
	ErrInvalidTrajectory = errors.New("motion/planner: invalid trajectory")
	// ErrInvalidParameters indicates configuration values are inconsistent.
	ErrInvalidParameters = errors.New("motion/planner: invalid parameters")
)

// State represents the robot pose and velocity at a specific timestamp.
type State struct {
	Position  vec.Vector3D
	Yaw       float32
	Speed     float32
	Timestamp float32
}

// Controls represents aggregate control commands issued to the robot.
type Controls struct {
	Linear  float32
	Angular float32
	Effort  vec.Vector
}

// Trajectory encapsulates future states the robot should follow and
// corresponding nominal controls.
type Trajectory struct {
	States   []State
	Controls []Controls
}

// Constraints defines the physical limits the planner must respect.
type Constraints struct {
	MaxSpeed               float32
	MaxAcceleration        float32
	MaxDeceleration        float32
	MaxJerk                float32
	MaxTurnRate            float32
	MaxTurnAcceleration    float32
	MaxLateralAcceleration float32
}

// PIDGains holds proportional-integral-derivative coefficients.
type PIDGains struct {
	P float32
	I float32
	D float32
}

// Parameters configures the planner tuning knobs.
type Parameters struct {
	SamplePeriod      float32
	LookaheadDistance float32
	SpeedPID          PIDGains
	HeadingPID        PIDGains
	LateralPID        PIDGains
}

func validateConstraints(c Constraints) error {
	if c.MaxSpeed <= 0 || c.MaxAcceleration <= 0 || c.MaxDeceleration <= 0 || c.MaxJerk <= 0 {
		return ErrInvalidParameters
	}
	if c.MaxTurnRate <= 0 || c.MaxTurnAcceleration <= 0 {
		return ErrInvalidParameters
	}
	if c.MaxLateralAcceleration <= 0 {
		return ErrInvalidParameters
	}
	return nil
}

func validateParameters(p Parameters) error {
	if p.SamplePeriod <= 0 || p.LookaheadDistance <= 0 {
		return ErrInvalidParameters
	}
	if !validGains(p.SpeedPID) || !validGains(p.HeadingPID) || !validGains(p.LateralPID) {
		return ErrInvalidParameters
	}
	return nil
}

func validGains(g PIDGains) bool {
	if stdmath.IsNaN(float64(g.P)) || stdmath.IsNaN(float64(g.I)) || stdmath.IsNaN(float64(g.D)) {
		return false
	}
	// Allow zero gains but not negative integrator.
	return !stdmath.IsInf(float64(g.P), 0) &&
		!stdmath.IsInf(float64(g.I), 0) &&
		!stdmath.IsInf(float64(g.D), 0)
}

func clampFloat(v, min, max float32) float32 {
	return ermath.Clamp(v, min, max)
}
