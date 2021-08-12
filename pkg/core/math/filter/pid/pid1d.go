package pid

import (
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type PID1D struct {
	P, I, D                          float32
	min, max                         float32
	input, lastInput, Output, Target float32
	iTerm                            float32
	samplePeriod                     float32
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

func (p *PID1D) Init(input float32) *PID1D {
	p.input = input
	p.lastInput = input
	p.iTerm = 0
	return p
}

func (p *PID1D) Update(input, samplePeriod float32) *PID1D {
	p.lastInput, p.input = p.input, input
	p.samplePeriod = samplePeriod
	return p
}

func (p *PID1D) Calculate() *PID1D {
	E := p.Target - p.input
	D := (p.input - p.lastInput)

	p.iTerm = math.Clamp(p.iTerm+p.I*E*p.samplePeriod, p.min, p.max)
	p.Output = math.Clamp(p.P*E+p.iTerm-p.D*D/p.samplePeriod, p.min, p.max)

	return p
}
