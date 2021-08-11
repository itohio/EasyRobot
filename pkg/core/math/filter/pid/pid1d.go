package pid

import (
	"time"

	"github.com/foxis/EasyRobot/pkg/core/math"
)

type PID1D struct {
	P, I, D                          float32
	min, max                         float32
	input, lastInput, Output, Target float32
	iTerm                            float32
	lastTimestamp, timestamp         time.Time
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
	p.timestamp = time.Now()
	p.lastTimestamp = p.timestamp
	p.iTerm = 0
	return p
}

func (p *PID1D) Update(input float32) *PID1D {
	p.lastInput, p.input = p.input, input
	p.lastTimestamp, p.timestamp = p.timestamp, time.Now()
	return p
}

func (p *PID1D) Calculate() *PID1D {
	timeChange := p.timestamp.Sub(p.lastTimestamp)
	delta := float32(timeChange.Seconds())
	if timeChange == 0 {
		delta = 1
	}

	E := p.Target - p.input
	D := (p.input - p.lastInput)

	p.iTerm = math.Clamp(p.iTerm+p.I*E*delta, p.min, p.max)
	p.Output = math.Clamp(p.P*E+p.iTerm-p.D*D/delta, p.min, p.max)

	return p
}
