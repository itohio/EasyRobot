package eskf

import "github.com/chewxy/math32"

// State is the nominal state: Q the attitude [w,x,y,z] (sensor to world,
// z up), V and P the world velocity (m/s) and position (m), GyroBias and
// AccelBias in the sensor frame. Its methods are the mechanisation: turn
// by a rate, move by a force, take a correction.
type State struct {
	Q         [4]float32
	V, P      [3]float32
	GyroBias  [3]float32
	AccelBias [3]float32

	r [3][3]float32 // Q as a matrix
}

// reset: the identity attitude, at rest at the origin, the gyro bias gb,
// no accelerometer bias.
func (s *State) reset(gb [3]float32) {
	s.Q, s.V, s.P = [4]float32{1, 0, 0, 0}, [3]float32{}, [3]float32{}
	s.GyroBias, s.AccelBias = gb, [3]float32{}
	s.refresh()
}

// refresh recomputes r from Q.
func (s *State) refresh() {
	w, x, y, z := s.Q[0], s.Q[1], s.Q[2], s.Q[3]
	s.r = [3][3]float32{
		{1 - 2*(y*y+z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)},
		{2 * (x*y + w*z), 1 - 2*(x*x+z*z), 2 * (y*z - w*x)},
		{2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x+y*y)},
	}
}

// normalise keeps Q a unit quaternion, and r with it.
func (s *State) normalise() {
	k := 1 / math32.Sqrt(s.Q[0]*s.Q[0]+s.Q[1]*s.Q[1]+s.Q[2]*s.Q[2]+s.Q[3]*s.Q[3])
	for i := range 4 {
		s.Q[i] *= k
	}
	s.refresh()
}

// turn steps the attitude by the body rate w (bias removed) over dt, an
// exact rotation by the angle θ = |w|·dt about w: q ← q ⊗ [cos(θ/2),
// sin(θ/2)·w/|w|], normalised against rounding. (The first-order step
// q ⊗ [1, w·dt/2] turns by 2·atan(θ/2), short by θ³/12: 0.1 rad at 10 rad/s
// over a minute.)
func (s *State) turn(w [3]float32, dt float32) {
	a := 0.5 * dt * math32.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]) // θ/2
	c, sa := 1-a*a/2, 1-a*a/6                                  // cos(θ/2), sin(θ/2)/(θ/2), to a⁴ below 1e-3
	if a >= 1e-3 {
		var sn float32
		sn, c = math32.Sincos(a)
		sa = sn / a
	}
	h := 0.5 * dt * sa
	q := s.Q
	s.Q = [4]float32{
		c*q[0] - h*(q[1]*w[0]+q[2]*w[1]+q[3]*w[2]),
		c*q[1] + h*(q[0]*w[0]+q[2]*w[2]-q[3]*w[1]),
		c*q[2] + h*(q[0]*w[1]-q[1]*w[2]+q[3]*w[0]),
		c*q[3] + h*(q[0]*w[2]+q[1]*w[1]-q[2]*w[0]),
	}
	s.normalise()
}

// move turns the force a (bias removed) into the world with the current
// attitude, takes gravity g off and integrates velocity and position over
// dt; it returns the world acceleration.
func (s *State) move(a [3]float32, g, dt float32) (aw [3]float32) {
	for i := range 3 {
		aw[i] = s.r[i][0]*a[0] + s.r[i][1]*a[1] + s.r[i][2]*a[2]
	}
	aw[2] -= g
	for i := range 3 {
		s.P[i] += s.V[i]*dt + 0.5*aw[i]*dt*dt
		s.V[i] += aw[i] * dt
	}
	return aw
}

// correct moves the error d into the nominal state: q ← [1, δθ/2] ⊗ q
// (the error is in the world), normalised; the others added.
func (s *State) correct(d *[n]float32) {
	h0, h1, h2 := 0.5*d[iTh], 0.5*d[iTh+1], 0.5*d[iTh+2]
	q := s.Q
	s.Q = [4]float32{
		q[0] - h0*q[1] - h1*q[2] - h2*q[3],
		q[1] + h0*q[0] + h1*q[3] - h2*q[2],
		q[2] - h0*q[3] + h1*q[0] + h2*q[1],
		q[3] + h0*q[2] - h1*q[1] + h2*q[0],
	}
	for i := range 3 {
		s.V[i] += d[iV+i]
		s.P[i] += d[iP+i]
		s.GyroBias[i] += d[iBg+i]
		s.AccelBias[i] += d[iBa+i]
	}
	s.normalise()
}
