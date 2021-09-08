package vaj

import "github.com/foxis/EasyRobot/pkg/core/math"

type VAJ1D struct {
	maxV, maxA, jerk       float32
	direction              float32
	velocity, acceleration float32
	Input, Output, Target  float32
}

func New1D(maxVelocity, maxAcceleration, jerk float32) VAJ1D {
	return VAJ1D{
		maxV: maxVelocity,
		maxA: maxAcceleration,
		jerk: jerk,
	}
}

func (l *VAJ1D) Reset() *VAJ1D {
	l.velocity = 0
	l.acceleration = 0
	l.direction = 1
	return l
}

func (l *VAJ1D) Update(samplePeriod float32) *VAJ1D {
	l.Output += (l.velocity + (.5*l.acceleration-l.jerk*samplePeriod/6)*samplePeriod) * samplePeriod

	var direction float32 = 1
	if (l.Output-l.Input)/(l.Target-l.Output) >= 0.5 {
		direction = -1
	}

	l.velocity = math.Clamp(
		l.velocity+l.acceleration*samplePeriod-.5*direction*l.jerk*samplePeriod*samplePeriod,
		-l.maxV, l.maxV,
	)
	l.acceleration = math.Clamp(
		l.acceleration+l.jerk*direction*samplePeriod,
		-l.maxA, l.maxA,
	)

	return l
}
