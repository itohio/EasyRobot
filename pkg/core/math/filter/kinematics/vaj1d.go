package vaj

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type VAJ1D struct {
	maxV, maxA, maxJ           float32
	v1max, v2max, vamax        float32
	Velocity, Acceleration, j0 float32
	Input, Output, Target      float32
}

func New1D(maxVelocity, maxAcceleration, jerk float32) VAJ1D {
	vamax := (maxAcceleration * maxAcceleration) / (jerk * 2)
	v1max := maxVelocity / 2
	v2max := maxVelocity / 2
	if v1max > vamax {
		v1max = vamax
	}
	if v2max < maxVelocity-vamax {
		v2max = maxVelocity - vamax
	}
	return VAJ1D{
		maxV:  maxVelocity,
		maxA:  maxAcceleration,
		maxJ:  jerk,
		v1max: v1max,
		v2max: v2max,
		vamax: vamax,
	}
}

func (l *VAJ1D) Reset() *VAJ1D {
	l.Velocity = 0
	l.Acceleration = 0
	return l
}

func (l *VAJ1D) Update(samplePeriod float32) *VAJ1D {
	defer func() {
		l.Input = l.Output
	}()

	x1 := l.Target - l.Input
	var c float32 = 1
	if x1 < 1 {
		x1 = -x1
		c = -1
	}

	if x1 < .001 {
		l.Output = l.Input
		l.j0 = 0
		l.Velocity = 0
		l.Acceleration = 0
		return l
	}

	x0 := l.calculateKinematics(samplePeriod, x1)
	l.Output = l.Input + x0*c

	return l
}

func (l *VAJ1D) calculateKinematics(samplePeriod, x1 float32) float32 {
	dt := samplePeriod
	v0 := l.Velocity
	a0 := l.Acceleration

	// check the stage of the acceleration/deceleration part
	x0, v0, a0, jC := l.calculateJerk(samplePeriod, x1, v0, a0, l.j0)

	// check if we are able to stop in time
	stopAt := l.calculateStoppingDistance(v0, a0)

	if stopAt <= x1 {
		if math32.Abs(v0) >= l.maxV-l.maxA*dt-0.5*l.maxJ*dt*dt-0.001 {
			a0 = 0
			jC = 0
			v0 = l.maxV
		}
	} else if jC == 0 || (a0 < l.maxA && a0 >= 0) {
		jC = -1
	}

	// integrate the step
	x0 += (v0 + (0.5*a0*dt+(1/6)*l.j0*dt)*dt) * dt
	l.Velocity += (a0 + .5*l.j0*dt) * dt
	a0 += l.j0 * dt
	l.Acceleration = math.Clamp(a0, -l.maxA, l.maxA)
	l.j0 = jC * l.maxJ

	return x0
}

func (l *VAJ1D) calculateJerk(dt, x1, v0, a0, j0 float32) (float32, float32, float32, float32) {
	var x0, jC float32
	v0x := v0 + a0*dt + .5*j0*dt*dt
	if v0x >= 0 && v0x <= l.v1max {
		if j0 == -l.maxJ {
			_, t := math.Quad(.5*j0, a0, v0-l.v1max, 1e-6)
			if t <= 2*dt {
				x0 = (v0 + (0.5*a0+(1/6)*j0*t)*t) * t
				v0 += (a0 + .5*j0*t) * t
				a0 += j0 * t
			}
		}
		jC = 1
	} else if v0x < l.maxV && v0x > l.v2max {
		if j0 == l.maxJ {
			t, _ := math.Quad(.5*j0, a0, v0-l.v2max, 1e-6)
			x0 = (v0 + (0.5*a0+(1/6)*j0*t)*t) * t
			v0 += (a0 + .5*j0*t) * t
			a0 += j0 * t
		}
		jC = -1
	} else {
		jC = 0
	}

	return x0, v0, a0, jC
}

func (l *VAJ1D) calculateStoppingDistance(v0, a0 float32) float32 {
	var (
		s, s1, s2, s3 float32
		v1m, v2m      float32
		v1, v2, a1    float32
	)

	// Remove acceleration
	if a0 > 0 {
		t := a0 / l.maxJ
		jt := .5 * l.maxJ * t
		s = (v0 + (.5*a0-(1/3)*jt)*t) * t
		v0 += (a0 - jt) * t
		a0 = 0
	}

	if a0 == 0 {
		v1m = math32.Min(v0/2, l.vamax)
		v2m = math32.Max(v0/2, v0-l.vamax)
	} else {
		v1m = l.v1max
		v2m = l.v2max
	}

	// decelerate
	if v0 > v2m {
		_, t := math.Quad(-.5*l.maxJ, a0, v0-v2m, 1e-6)
		v2 = v2m
		v1 = v2m
		s1 = (v0 + (0.5*a0-(1/6)*l.maxJ*t)*t) * t
		a1 = a0 - l.maxJ*t
	} else {
		v1 = v0
		v2 = v0
		a1 = a0
	}

	// coast
	if v1 <= v2m && v1 > v1m {
		t := (v1 - v1m) / (-a1)
		v2 = v1m
		s2 = (v1 + .5*a1*t) * t
	}

	// remove deceleration
	if v2 > 0 {
		t, _ := math.Quad(.5*l.maxJ, a1, v2, 1e-6)
		s3 = (v2 + (.5*a1+(1/6)*l.maxJ*t)*t) * t
	}

	return s + s1 + s2 + s3
}
