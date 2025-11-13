package steer6

import (
	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/pkg/core/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/pkg/core/math/control/kinematics/wheels/internal/rigid"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	mattype "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const eps = 1e-6

type Config struct {
	WheelRadius  float32
	Track        float32
	FrontOffset  float32
	MiddleOffset float32
	RearOffset   float32
}

type drive struct {
	radius     float32
	track      float32
	halfTrack  float32
	frontX     float32
	middleX    float32
	rearX      float32
	params     [10]float32
	state      [6]float32
	positionsV []vec.Vector2D

	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

var _ kintypes.Bidirectional = (*drive)(nil)

func New(cfg Config) *drive {
	if cfg.WheelRadius <= 0 {
		panic("steer6: WheelRadius must be > 0")
	}
	if cfg.Track == 0 {
		panic("steer6: Track must be non-zero")
	}
	d := &drive{
		radius:    cfg.WheelRadius,
		track:     cfg.Track,
		halfTrack: cfg.Track * 0.5,
		frontX:    cfg.FrontOffset,
		middleX:   cfg.MiddleOffset,
		rearX:     -cfg.RearOffset,
		dimensions: kintypes.Dimensions{
			StateRows:    10,
			StateCols:    1,
			ControlSize:  10,
			ActuatorSize: 10,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:       false,
			Omnidirectional: false,
			ConstraintRank:  6,
		},
	}
	d.positionsV = []vec.Vector2D{
		{d.frontX, d.halfTrack},
		{d.frontX, -d.halfTrack},
		{d.middleX, d.halfTrack},
		{d.middleX, -d.halfTrack},
		{d.rearX, d.halfTrack},
		{d.rearX, -d.halfTrack},
	}
	return d
}

func (d *drive) Dimensions() kintypes.Dimensions {
	return d.dimensions
}

func (d *drive) Capabilities() kintypes.Capabilities {
	return d.capabilities
}

func (*drive) ConstraintSet() kintypes.Constraints {
	return kintypes.Constraints{}
}

func (d *drive) Params() vec.Vector {
	return d.params[:]
}

func (d *drive) Effector() vec.Vector {
	return d.state[:]
}

// Forward reads the 10×1 `state` column `[ω_fl, ω_fr, ω_ml, ω_mr, ω_rl, ω_rr,
// δ_fl, δ_fr, δ_rl, δ_rr]` and writes `[v, ω, δ_fl, δ_fr, δ_rl, δ_rr]` into
// `destination`.
func (d *drive) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(state, len(d.params)); err != nil {
		return err
	}
	if err := ensureColumn(destination, len(d.state)); err != nil {
		return err
	}

	stateView := state.View().(mat.Matrix)
	for i := range d.params {
		d.params[i] = stateView[i][0]
	}

	headings := []float32{
		d.params[6],
		d.params[7],
		0,
		0,
		d.params[8],
		d.params[9],
	}
	v, omega := rigid.SolveTwist(d.radius, d.params[:6], headings, d.positionsV)
	d.state[0] = v
	d.state[1] = omega
	d.state[2] = d.params[6]
	d.state[3] = d.params[7]
	d.state[4] = d.params[8]
	d.state[5] = d.params[9]

	destView := destination.View().(mat.Matrix)
	for i := range d.state {
		destView[i][0] = d.state[i]
	}
	return nil
}

// Backward interprets `destination` as `[v, ω, δ_fl, δ_fr, δ_rl, δ_rr]` and
// writes wheel rates plus steering angles into `controls` using the same order
// as the state vector.
func (d *drive) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(destination, len(d.state)); err != nil {
		return err
	}
	if err := ensureColumn(controls, len(d.params)); err != nil {
		return err
	}
	if state != nil {
		if err := ensureColumn(state, len(d.params)); err != nil {
			return err
		}
		stateView := state.View().(mat.Matrix)
		for i := range d.params {
			d.params[i] = stateView[i][0]
		}
	}

	destView := destination.View().(mat.Matrix)
	for i := range d.state {
		d.state[i] = destView[i][0]
	}

	v := d.state[0]
	omega := d.state[1]
	angles := d.steeringFor(v, omega)
	d.state[2] = angles[0]
	d.state[3] = angles[1]
	d.state[4] = angles[2]
	d.state[5] = angles[3]
	d.params[6] = angles[0]
	d.params[7] = angles[1]
	d.params[8] = angles[2]
	d.params[9] = angles[3]

	rigid.AssignWheelRates(d.radius, d.params[:6], v, omega, []float32{
		angles[0],
		angles[1],
		0,
		0,
		angles[2],
		angles[3],
	}, d.positionsV)

	controlView := controls.View().(mat.Matrix)
	for i := range d.params {
		controlView[i][0] = d.params[i]
	}
	return nil
}

func (d *drive) steeringFor(v, omega float32) []float32 {
	if math32.Abs(omega) < eps || math32.Abs(v) < eps {
		return []float32{0, 0, 0, 0}
	}
	r := v / omega
	return []float32{
		steerAngle(d.frontX, d.halfTrack, r),
		steerAngle(d.frontX, -d.halfTrack, r),
		steerAngle(d.rearX, d.halfTrack, r),
		steerAngle(d.rearX, -d.halfTrack, r),
	}
}

func steerAngle(x, y, radius float32) float32 {
	den := radius - y
	if math32.Abs(den) < eps {
		den = math32.Copysign(eps, den)
	}
	return math32.Atan(x / den)
}

func ensureColumn(m mattype.Matrix, rows int) error {
	if m == nil {
		return kintypes.ErrInvalidDimensions
	}
	if m.Rows() != rows || m.Cols() < 1 {
		return kintypes.ErrInvalidDimensions
	}
	return nil
}
