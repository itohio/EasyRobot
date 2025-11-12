package steer6

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/control/kinematics"
	"github.com/itohio/EasyRobot/pkg/core/math/control/kinematics/wheels/internal/rigid"
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
}

var _ kinematics.Kinematics = (*drive)(nil)

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

func (*drive) DOF() int {
	return 10
}

func (d *drive) Params() vec.Vector {
	return d.params[:]
}

func (d *drive) Effector() vec.Vector {
	return d.state[:]
}

func (d *drive) Forward() bool {
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
	return true
}

func (d *drive) Inverse() bool {
	v := d.state[0]
	omega := d.state[1]
	angles := d.steeringFor(v, omega)
	copy(d.state[2:], angles)
	copy(d.params[6:], angles)
	rigid.AssignWheelRates(d.radius, d.params[:6], v, omega, []float32{
		angles[0],
		angles[1],
		0,
		0,
		angles[2],
		angles[3],
	}, d.positionsV)
	return true
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
