package pid

import (
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type PID struct {
	P, I, D                          vec.Vector
	min, max                         vec.Vector
	Input, lastInput, Output, Target vec.Vector
	iTerm                            vec.Vector
}

func New(p, i, d, min, max vec.Vector) PID {
	N := len(p)
	if N != len(i) || N != len(d) || N != len(min) || N != len(max) {
		panic(-1)
	}
	return PID{
		P:      p,
		I:      i,
		D:      d,
		min:    min,
		max:    max,
		Input:  vec.New(N),
		Output: vec.New(N),
		Target: vec.New(N),
	}
}

func (p *PID) Reset() *PID {
	copy(p.lastInput, p.Input)
	p.iTerm.FillC(0)
	return p
}

func (p *PID) Update(samplePeriod float32) *PID {

	E := p.Target.Clone().Sub(p.Input)
	D := p.Input.Clone().Sub(p.lastInput)

	p.iTerm.Add(p.I.Clone().Multiply(E).MulC(samplePeriod)).Clamp(p.min, p.max)
	p.Output.CopyFrom(0, p.P.Clone().Multiply(E).Add(p.iTerm).Sub(p.D.Clone().Multiply(D).DivC(samplePeriod)).Clamp(p.min, p.max))

	copy(p.lastInput, p.Input)
	return p
}
