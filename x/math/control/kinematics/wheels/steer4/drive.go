package steer4

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
	WheelRadius float32
	Wheelbase   float32
	Track       float32
	FrontDrive  bool
	RearDrive   bool
}

type drive struct {
	radius     float32
	wheelbase  float32
	track      float32
	halfBase   float32
	halfTrack  float32
	frontDrive bool
	rearDrive  bool

	params [6]float32
	state  [4]float32
	posV   []vec.Vector2D

	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

var _ kintypes.Bidirectional = (*drive)(nil)

func New(cfg Config) *drive {
	if cfg.WheelRadius <= 0 {
		panic("steer4: WheelRadius must be > 0")
	}
	if cfg.Track == 0 {
		panic("steer4: Track must be non-zero")
	}
	d := &drive{
		radius:     cfg.WheelRadius,
		wheelbase:  cfg.Wheelbase,
		track:      cfg.Track,
		halfBase:   cfg.Wheelbase * 0.5,
		halfTrack:  cfg.Track * 0.5,
		frontDrive: cfg.FrontDrive,
		rearDrive:  cfg.RearDrive,
		dimensions: kintypes.Dimensions{
			StateRows:    6,
			StateCols:    1,
			ControlSize:  6,
			ActuatorSize: 6,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:       false,
			Omnidirectional: false,
			ConstraintRank:  4,
		},
	}
	d.posV = []vec.Vector2D{
		{d.halfBase, d.halfTrack},
		{d.halfBase, -d.halfTrack},
		{-d.halfBase, d.halfTrack},
		{-d.halfBase, -d.halfTrack},
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

// Forward reads the 6×1 `state` column `[ω_fl, ω_fr, ω_rl, ω_rr, δ_fl, δ_fr]`
// and writes chassis velocity `[v, ω, δ_fl, δ_fr]` into `destination`.
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

	headings := []float32{d.params[4], d.params[5], 0, 0}
	v, omega := rigid.SolveTwist(d.radius, d.params[:4], headings, d.posV)
	d.state[0] = v
	d.state[1] = omega
	d.state[2] = d.params[4]
	d.state[3] = d.params[5]

	destView := destination.View().(mat.Matrix)
	for i := range d.state {
		destView[i][0] = d.state[i]
	}
	return nil
}

// Backward interprets `destination` as `[v, ω, δ_fl, δ_fr]` and writes solved
// wheel rates and steering angles back into the `controls` column vector in the
// same order as `state`.
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
	dl, dr := d.steeringFor(v, omega)
	d.state[2], d.state[3] = dl, dr
	d.params[4], d.params[5] = dl, dr
	rigid.AssignWheelRates(d.radius, d.params[:4], v, omega, []float32{dl, dr, 0, 0}, d.posV)

	controlView := controls.View().(mat.Matrix)
	for i := range d.params {
		controlView[i][0] = d.params[i]
	}
	return nil
}

func (d *drive) steeringFor(v, omega float32) (float32, float32) {
	if math32.Abs(omega) < eps || math32.Abs(v) < eps {
		return 0, 0
	}
	radius := math32.Abs(v / omega)
	if radius < d.halfTrack {
		radius = d.halfTrack
	}
	inner := math32.Atan(d.wheelbase / (radius - d.halfTrack))
	outer := math32.Atan(d.wheelbase / (radius + d.halfTrack))
	if omega > 0 {
		return inner, outer
	}
	return -outer, -inner
}

func (d *drive) positions() []vec.Vector2D {
	return d.posV
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
