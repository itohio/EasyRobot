package pid

import (
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type PID struct {
	P, I, D                          vec.Vector
	min, max                         vec.Vector
	input, lastInput, Output, Target vec.Vector
	iTerm                            vec.Vector
	samplePeriod                     float32
}

func New(p, i, d, min, max vec.Vector) PID {
	N := len(p)
	if N != len(i) || N != len(d) || N != len(min) || N != len(max) {
		panic(-1)
	}
	return PID{
		P:   p,
		I:   i,
		D:   d,
		min: min,
		max: max,
	}
}

func (p *PID) Init(input vec.Vector) *PID {
	if len(p.P) != len(input) {
		panic(-1)
	}
	p.input = input
	p.lastInput = input
	p.iTerm.FillC(0)
	return p
}

func (p *PID) Update(input vec.Vector, samplePeriod float32) *PID {
	if len(p.P) != len(input) {
		panic(-1)
	}
	p.lastInput, p.input = p.input, input
	p.samplePeriod = samplePeriod
	return p
}

func (p *PID) Calculate() *PID {
	E := p.Target.Clone().Sub(p.input)
	D := p.input.Clone().Sub(p.lastInput)

	p.iTerm.Add(p.I.Clone().Multiply(E).MulC(p.samplePeriod)).Clamp(p.min, p.max)
	p.Output.CopyFrom(0, p.P.Clone().Multiply(E).Add(p.iTerm).Sub(p.D.Clone().Multiply(D).DivC(p.samplePeriod)).Clamp(p.min, p.max))

	return p
}
