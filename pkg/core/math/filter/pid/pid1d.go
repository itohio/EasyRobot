package pid

import (
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type PID1D struct {
	P, I, D               float32
	min, max              float32
	Input, Output, Target float32
	lastInput, iTerm      float32
}

func New1D(p, i, d, min, max float32) PID1D {
	return PID1D{
		P:   p,
		I:   i,
		D:   d,
		min: min,
		max: max,
	}
}

func (p *PID1D) Reset() *PID1D {
	p.lastInput = p.Input
	p.iTerm = 0
	return p
}

func (p *PID1D) Update(samplePeriod float32) *PID1D {
	E := p.Target - p.Input
	D := p.Input - p.lastInput

	p.iTerm = math.Clamp(p.iTerm+p.I*E*samplePeriod, p.min, p.max)
	p.Output = math.Clamp(p.P*E+p.iTerm-p.D*D/samplePeriod, p.min, p.max)

	p.lastInput = p.Input
	return p
}
