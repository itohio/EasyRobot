// Package chain declares serial joint chains in model units (radians, metres),
// maps them to servo degrees, validates them, and builds the matching kinematic
// model: the analytic coxa–femur–tibia leg (planar.New3DOF) when the chain has
// that shape, damped least-squares DH (dh.New) otherwise. The legged layer
// (legged.go) mounts leg chains on a body and solves all legs per tick without
// allocation.
//
// Degrees appear only in ServoMap: servoDeg = ZeroDeg + Sign·Ratio·deg(q).
package chain

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/joints/dh"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/joints/planar"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// JointKind selects the DH parameter a joint drives.
type JointKind uint8

const (
	// Revolute joints drive θ: θ = Offset + q (radians).
	Revolute JointKind = iota
	// Prismatic joints drive d: d = D + q (metres); θ = Offset is constant.
	Prismatic
)

// Default servo travel used when ServoMap.MinDeg and MaxDeg are both zero.
const (
	DefaultServoMinDeg = 0
	DefaultServoMaxDeg = 180
)

const (
	radToDeg = 180 / math32.Pi
	degToRad = math32.Pi / 180
	// shapeTol is the tolerance on α when matching the coxa–femur–tibia shape.
	shapeTol = 1e-6
)

// ServoMap maps a joint value to a servo command in degrees:
// servoDeg = ZeroDeg + Sign·Ratio·deg(q), deg(q) = q·180/π. For prismatic
// joints (q in metres) Ratio carries the metres→servo-degree gearing.
type ServoMap struct {
	Channel int     // PWM / PCA9685 channel, for the driver; not used here
	ZeroDeg float32 // servo angle when q == 0 (calibration)
	Sign    int8    // +1 or −1 (mirror-mounted servos)
	Ratio   float32 // horn/gear ratio, > 0
	// MinDeg, MaxDeg is the servo travel. Both zero means the default 0…180°.
	MinDeg, MaxDeg float32
}

// Range returns the servo travel in degrees (the 0…180° default when MinDeg
// and MaxDeg are both zero).
func (s ServoMap) Range() (lo, hi float32) {
	if s.MinDeg == 0 && s.MaxDeg == 0 {
		return DefaultServoMinDeg, DefaultServoMaxDeg
	}
	return s.MinDeg, s.MaxDeg
}

func (s ServoMap) toDeg(q float32) float32 {
	return s.ZeroDeg + float32(s.Sign)*s.Ratio*q*radToDeg
}

func (s ServoMap) fromDeg(d float32) float32 {
	return (d - s.ZeroDeg) / (float32(s.Sign) * s.Ratio) * degToRad
}

// Joint is one standard (distal) DH row, A = Rz(θ)·Tz(d)·Tx(A)·Rx(Alpha), in
// radians and metres, plus limits, rate limits and the servo mapping.
type Joint struct {
	Name                string
	Kind                JointKind
	A, Alpha, D, Offset float32 // DH r, α, d and the constant θ offset
	Min, Max            float32 // joint limits in model units; Min < Max required
	MaxVel, MaxAcc      float32 // joint rate limits (rad/s, rad/s²); 0 = unspecified
	Servo               ServoMap
}

// Chain is a serial chain, base to tip.
type Chain struct {
	Name   string
	Joints []Joint
}

var (
	// ErrLimits reports non-finite joint limits or Min >= Max (the zero value
	// would pin the joint at 0).
	ErrLimits = errors.New("chain: joint limits must be finite with Min < Max")
	// ErrServoMap reports a malformed ServoMap: |Sign| != 1, Ratio <= 0,
	// non-finite fields or an inverted servo travel.
	ErrServoMap = errors.New("chain: invalid servo map")
	// ErrServoRange reports joint limits that map outside the servo travel
	// (Validate) or a servo command clamped to the travel (ToServo).
	ErrServoRange = errors.New("chain: servo angle outside the servo travel")
	// ErrZeroPose reports a zero-pose foot position that differs from the
	// measured one: a sign, offset or length mistake.
	ErrZeroPose = errors.New("chain: zero pose does not match")
)

// Deg converts degrees to radians, for declaring limits.
func Deg(d float32) float32 { return d * degToRad }

func finite(vs ...float32) bool {
	for _, v := range vs {
		if math32.IsNaN(v) || math32.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// Validate checks every joint: supported Kind, finite geometry, finite limits
// with Min < Max, non-negative finite rate limits, |Sign| == 1, Ratio > 0, a
// valid servo travel, and that [Min, Max] maps inside the servo travel. Every
// error wraps kintypes.ErrInvalidConfig and, where it applies, ErrLimits,
// ErrServoMap or ErrServoRange.
func (c Chain) Validate() error {
	if len(c.Joints) == 0 {
		return fmt.Errorf("%w: chain %q: no joints", kintypes.ErrInvalidConfig, c.Name)
	}
	for i, j := range c.Joints {
		if err := j.validate(); err != nil {
			return fmt.Errorf("chain %q joint %d (%s): %w", c.Name, i, j.Name, err)
		}
	}
	return nil
}

func (j Joint) validate() error {
	if j.Kind != Revolute && j.Kind != Prismatic {
		return fmt.Errorf("%w: unsupported kind %d", kintypes.ErrInvalidConfig, j.Kind)
	}
	if !finite(j.A, j.Alpha, j.D, j.Offset) {
		return fmt.Errorf("%w: non-finite DH parameter", kintypes.ErrInvalidConfig)
	}
	if !finite(j.Min, j.Max) || !(j.Min < j.Max) {
		return fmt.Errorf("%w: %w: [%g, %g]", kintypes.ErrInvalidConfig, ErrLimits, j.Min, j.Max)
	}
	if !finite(j.MaxVel, j.MaxAcc) || j.MaxVel < 0 || j.MaxAcc < 0 {
		return fmt.Errorf("%w: rate limits must be finite and >= 0, got %g, %g", kintypes.ErrInvalidConfig, j.MaxVel, j.MaxAcc)
	}
	s := j.Servo
	lo, hi := s.Range()
	if (s.Sign != 1 && s.Sign != -1) || !(s.Ratio > 0) || !finite(s.Ratio, s.ZeroDeg, lo, hi) || !(lo < hi) {
		return fmt.Errorf("%w: %w: %+v", kintypes.ErrInvalidConfig, ErrServoMap, s)
	}
	a, b := s.toDeg(j.Min), s.toDeg(j.Max)
	if a > b {
		a, b = b, a
	}
	if a < lo || b > hi {
		return fmt.Errorf("%w: %w: limits map to [%g°, %g°], travel [%g°, %g°]", kintypes.ErrInvalidConfig, ErrServoRange, a, b, lo, hi)
	}
	return nil
}

// DH returns the validated standard DH table: Theta = Offset, R = A, D, Alpha,
// Index 0 (θ) for revolute and 3 (d) for prismatic joints, limits copied.
func (c Chain) DH() ([]dh.Config, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	cfg := make([]dh.Config, len(c.Joints))
	for i, j := range c.Joints {
		cfg[i] = dh.Config{Min: j.Min, Max: j.Max, Theta: j.Offset, Alpha: j.Alpha, R: j.A, D: j.D}
		if j.Kind == Prismatic {
			cfg[i].Index = 3
		}
	}
	return cfg, nil
}

// Planar3 returns the planar.New3DOF configuration when the chain is a
// coxa–femur–tibia leg: three revolute joints, α = (π/2, 0, 0), all d and θ
// offsets zero. Then the planar and DH models are the same mechanism. Other
// shapes return kintypes.ErrUnsupportedOperation; invalid chains, including a
// leg-shaped chain without Coxa >= 0 and Femur, Tibia > 0, return
// kintypes.ErrInvalidConfig.
func (c Chain) Planar3() ([3]planar.Config, error) {
	var out [3]planar.Config
	if err := c.Validate(); err != nil {
		return out, err
	}
	if err := c.legShape(); err != nil {
		return out, err
	}
	for i, j := range c.Joints {
		out[i] = planar.Config{Min: j.Min, Max: j.Max, Length: j.A}
	}
	return out, nil
}

func (c Chain) legShape() error {
	if len(c.Joints) != 3 {
		return fmt.Errorf("%w: chain %q: %d joints, a leg has 3", kintypes.ErrUnsupportedOperation, c.Name, len(c.Joints))
	}
	alpha := [3]float32{math32.Pi / 2, 0, 0}
	for i, j := range c.Joints {
		if j.Kind != Revolute || math32.Abs(j.Alpha-alpha[i]) > shapeTol || j.D != 0 || j.Offset != 0 {
			return fmt.Errorf("%w: chain %q joint %d is not coxa–femur–tibia shaped", kintypes.ErrUnsupportedOperation, c.Name, i)
		}
	}
	// The joints say leg: bad lengths are a configuration error, not a
	// reason to fall back to another solver.
	if l := c.Joints; !(l[0].A >= 0 && l[1].A > 0 && l[2].A > 0) {
		return fmt.Errorf("%w: chain %q: leg lengths need Coxa >= 0, Femur and Tibia > 0, got %g, %g, %g", kintypes.ErrInvalidConfig, c.Name, l[0].A, l[1].A, l[2].A)
	}
	return nil
}

// Option tunes the DH solver built by Model when the chain is not a leg.
type Option func(*modelOptions)

type modelOptions struct {
	eps           float32
	maxIterations int
}

func defaultOptions(opts []Option) modelOptions {
	o := modelOptions{eps: 1e-4, maxIterations: 100}
	for _, f := range opts {
		f(&o)
	}
	return o
}

// WithTolerance sets the DH position tolerance in metres (default 1e-4).
func WithTolerance(eps float32) Option { return func(o *modelOptions) { o.eps = eps } }

// WithMaxIterations sets the DH iteration budget (default 100).
func WithMaxIterations(n int) Option { return func(o *modelOptions) { o.maxIterations = n } }

// Model validates the chain and returns the analytic coxa–femur–tibia solver
// (planar.New3DOF; Forward writes [x y z roll pitch yaw]) when Planar3 matches,
// otherwise DH damped least squares (dh.New; Forward writes
// [x y z qx qy qz qw]). Both read the target position from rows 0–2. A chain
// Planar3 rejects with ErrInvalidConfig (e.g. leg-shaped with a femur length
// of 0) is an error, not a DH chain.
func (c Chain) Model(opts ...Option) (kintypes.Bidirectional, error) {
	m, _, err := c.model(defaultOptions(opts))
	return m, err
}

// effector rows written by Forward and read by Backward.
const (
	planarRows = 6
	dhRows     = 7
)

func (c Chain) model(o modelOptions) (kintypes.Bidirectional, int, error) {
	if p3, err := c.Planar3(); err == nil {
		m, err := planar.New3DOF(p3)
		if err != nil {
			return nil, 0, err
		}
		return m, planarRows, nil
	} else if !errors.Is(err, kintypes.ErrUnsupportedOperation) {
		return nil, 0, err
	}
	cfg, err := c.DH()
	if err != nil {
		return nil, 0, err
	}
	m, err := dh.New(o.eps, o.maxIterations, cfg...)
	if err != nil {
		return nil, 0, err
	}
	return m, dhRows, nil
}

// ToServo writes servo degrees for joint values q (len(q) == len(servoDeg) ==
// number of joints; otherwise kintypes.ErrInvalidDimensions). A joint outside
// its limits (NaN counts as below Min) is clamped first and the call returns
// kintypes.ErrJointLimit; a command outside the servo travel (only possible
// for a chain that does not Validate) is clamped and returns ErrServoRange.
// Every entry is written either way; the first error is returned. No
// allocation.
func (c Chain) ToServo(q, servoDeg []float32) error {
	if len(q) != len(c.Joints) || len(servoDeg) != len(c.Joints) {
		return kintypes.ErrInvalidDimensions
	}
	var err error
	for i := range c.Joints {
		j := &c.Joints[i]
		v := q[i]
		switch {
		case !(v >= j.Min):
			v = j.Min
			err = firstErr(err, kintypes.ErrJointLimit)
		case v > j.Max:
			v = j.Max
			err = firstErr(err, kintypes.ErrJointLimit)
		}
		d := j.Servo.toDeg(v)
		lo, hi := j.Servo.Range()
		switch {
		case d < lo:
			d = lo
			err = firstErr(err, ErrServoRange)
		case d > hi:
			d = hi
			err = firstErr(err, ErrServoRange)
		}
		servoDeg[i] = d
	}
	return err
}

// FromServo converts servo degrees back to joint values (the inverse of
// ToServo). Values are written unclamped; a value outside the joint limits
// returns kintypes.ErrJointLimit. No allocation.
func (c Chain) FromServo(servoDeg, q []float32) error {
	if len(q) != len(c.Joints) || len(servoDeg) != len(c.Joints) {
		return kintypes.ErrInvalidDimensions
	}
	var err error
	for i := range c.Joints {
		j := &c.Joints[i]
		v := j.Servo.fromDeg(servoDeg[i])
		if !(v >= j.Min && v <= j.Max) {
			err = firstErr(err, kintypes.ErrJointLimit)
		}
		q[i] = v
	}
	return err
}

func firstErr(have, next error) error {
	if have != nil {
		return have
	}
	return next
}

// CheckZeroPose compares the tip position at q = 0 (all servos at ZeroDeg),
// computed by unclamped standard-DH FK, with a measured position `want` in the
// chain base frame. A distance above tol returns an error wrapping ErrZeroPose,
// which catches sign, offset and length mistakes before the first move.
func (c Chain) CheckZeroPose(want vec.Vector3D, tol float32) error {
	got := c.tip(nil)
	dx, dy, dz := got[0]-want[0], got[1]-want[1], got[2]-want[2]
	if d := math32.Sqrt(dx*dx + dy*dy + dz*dz); !(d <= tol) {
		return fmt.Errorf("%w: chain %q: FK(0) = %v, measured %v, |Δ| = %g m > %g m", ErrZeroPose, c.Name, got, want, d, tol)
	}
	return nil
}

// tip is unclamped standard-DH FK of the tip position; q == nil means all zero.
func (c Chain) tip(q []float32) vec.Vector3D {
	// T accumulates the base→frame transform as rotation r and translation p.
	r := [3][3]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	var p [3]float32
	for i, j := range c.Joints {
		var v float32
		if q != nil {
			v = q[i]
		}
		theta, d := j.Offset, j.D
		if j.Kind == Prismatic {
			d += v
		} else {
			theta += v
		}
		ct, st := math32.Cos(theta), math32.Sin(theta)
		ca, sa := math32.Cos(j.Alpha), math32.Sin(j.Alpha)
		a := [3][3]float32{{ct, -st * ca, st * sa}, {st, ct * ca, -ct * sa}, {0, sa, ca}}
		t := [3]float32{j.A * ct, j.A * st, d}
		var nr [3][3]float32
		for row := range 3 {
			for k := range 3 {
				p[row] += r[row][k] * t[k]
			}
			for col := range 3 {
				for k := range 3 {
					nr[row][col] += r[row][k] * a[k][col]
				}
			}
		}
		r = nr
	}
	return vec.Vector3D(p)
}
