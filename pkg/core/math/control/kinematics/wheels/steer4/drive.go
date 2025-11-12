package steer4

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/control/kinematics"
	"github.com/itohio/EasyRobot/pkg/core/math/control/kinematics/wheels/internal/rigid"
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
}

var _ kinematics.Kinematics = (*drive)(nil)

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
	}
	d.posV = []vec.Vector2D{
		{d.halfBase, d.halfTrack},
		{d.halfBase, -d.halfTrack},
		{-d.halfBase, d.halfTrack},
		{-d.halfBase, -d.halfTrack},
	}
	return d
}

func (*drive) DOF() int {
	return 6
}

func (d *drive) Params() vec.Vector {
	return d.params[:]
}

func (d *drive) Effector() vec.Vector {
	return d.state[:]
}

func (d *drive) Forward() bool {
	headings := []float32{d.params[4], d.params[5], 0, 0}
	v, omega := rigid.SolveTwist(d.radius, d.params[:4], headings, d.posV)
	d.state[0] = v
	d.state[1] = omega
	d.state[2] = d.params[4]
	d.state[3] = d.params[5]
	return true
}

func (d *drive) Inverse() bool {
	v := d.state[0]
	omega := d.state[1]
	dl, dr := d.steeringFor(v, omega)
	d.state[2], d.state[3] = dl, dr
	d.params[4], d.params[5] = dl, dr
	rigid.AssignWheelRates(d.radius, d.params[:4], v, omega, []float32{dl, dr, 0, 0}, d.posV)
	return true
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
