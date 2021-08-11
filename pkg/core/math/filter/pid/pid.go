package pid

import (
	"time"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type PID struct {
	P, I, D                          vec.Vector
	min, max                         vec.Vector
	input, lastInput, Output, Target vec.Vector
	iTerm                            vec.Vector
	lastTimestamp, timestamp         time.Time
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
	p.timestamp = time.Now()
	p.lastTimestamp = p.timestamp
	p.iTerm.FillC(0)
	return p
}

func (p *PID) Update(input vec.Vector) *PID {
	if len(p.P) != len(input) {
		panic(-1)
	}
	p.lastInput, p.input = p.input, input
	p.lastTimestamp, p.timestamp = p.timestamp, time.Now()
	return p
}

func (p *PID) Calculate() *PID {
	timeChange := p.timestamp.Sub(p.lastTimestamp)
	delta := float32(timeChange.Seconds())
	if timeChange == 0 {
		delta = 1
	}

	E := p.Target.Clone().Sub(p.input)
	D := p.input.Clone().Sub(p.lastInput)

	p.iTerm.Add(p.I.Clone().Product(E).MulC(delta)).Clamp(p.min, p.max)
	p.Output.CopyFrom(0, p.P.Clone().Product(E).Add(p.iTerm).Sub(p.D.Clone().Product(D).DivC(delta)).Clamp(p.min, p.max))

	return p
}
