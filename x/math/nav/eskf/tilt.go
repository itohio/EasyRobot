package eskf

import "github.com/chewxy/math32"

// tiltConsider: the velocity, the position and the accelerometer's bias are
// consider states in the tilt update. Gravity's direction says nothing of
// the velocity or the position, which stay the integral of the corrected
// acceleration (corrected through their correlation with the attitude, one
// update moved the position by 9.9 m with exact samples); nor can it tell
// the bias from a horizontal acceleration, so the bias stays the
// calibration's (without its row in H the correlation alone still moved it).
const tiltConsider uint16 = 0x1ff&^0x7 | 0x7<<iBa // δv, δp, δb_a

// tiltModel is the tilt measurement: the force turned into the world,
// u = R(f − b_a)/|·|, against ẑ, gravity's direction; to first order
// u_x = −δθ_y and u_y = δθ_x. It measures the attitude only: the
// accelerometer's bias is the calibration's (Prior), since gravity's
// direction can't tell a bias from a horizontal acceleration (with the bias
// in H, a hand's strokes and a shake moved it by up to 1 m/s², and the
// attitude settled 4–7° off the force at rest). Its noise, never gated and
// never switched: TiltNoise; the force's magnitude's distance from g as a
// fraction of g (an acceleration along gravity); and the body's own
// acceleration, lin (the force's variance over the last half second above the
// sensor's noise, motionPower) times LinAccelTime as a white-equivalent
// density over the span since the last update, which one across gravity
// barely shows in the magnitude (a²/2g). On a desk that last term is zero
// and the tilt corrects the attitude quickly; while the body moves it
// stretches the correction's time scale well past a stroke, the gyro
// carries the attitude, and a stroke's acceleration and deceleration
// cancel. It returns the two rows of H, their innovations and the noise; ok
// false for a zero force.
func tiltModel(r *[3][3]float32, ba, acc [3]float32, c *Config, span, lin float32) (rows [2]row, ys [2]float32, rv float32, ok bool) {
	var a, u [3]float32
	for i := range 3 {
		a[i] = acc[i] - ba[i]
	}
	for i := range 3 {
		u[i] = r[i][0]*a[0] + r[i][1]*a[1] + r[i][2]*a[2]
	}
	g := math32.Sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
	if !(g > 0) {
		return rows, ys, 0, false
	}
	ig := 1 / g
	e := (g - c.Gravity) / c.Gravity
	rv = c.TiltNoise*c.TiltNoise + e*e
	if span > 0 {
		rv += c.LinAccelTime * lin / (span * c.Gravity * c.Gravity) // the body's acceleration over the span, white-equivalent
	}
	rows[0] = row{idx: [4]int{iTh + 1}, val: [4]float32{-1}, m: 1}
	rows[1] = row{idx: [4]int{iTh}, val: [4]float32{1}, m: 1}
	ys = [2]float32{u[0] * ig, u[1] * ig}
	return rows, ys, rv, true
}

// motionPower tracks the force's variance over about the last half second
// (motionWindow): the
// body's own acceleration plus the sensor's white noise. Its excess over the
// noise is what the tilt can't tell from a lean, so it scales the tilt's
// noise (continuously: a steady force on a desk adds nothing, a moving hand
// lengthens the tilt correction's time scale well past a stroke and the
// gyro carries the attitude through it).
// motionWindow is the span of motionPower's averages: short enough that the
// noise falls within a second of a set-down (1 s left the attitude 0.56° off
// the force 2 s after one, at LinAccelTime 16), long enough to hold a slow
// stroke's own acceleration (0.25 s dropped it between the stroke's peaks:
// 3 cm over 2 s read 57 %).
const motionWindow = 0.5 // s

type motionPower struct {
	mean [3]float32 // the force averaged over motionWindow
	v    float32    // its variance about that average, all axes, m²/s⁴
}

// reset starts the statistics at the force acc.
func (m *motionPower) reset(acc [3]float32) {
	m.mean, m.v = acc, 0
}

// add takes one sample over dt.
func (m *motionPower) add(acc [3]float32, dt float32) {
	k := min(dt/motionWindow, 1)
	var d2 float32
	for i := range 3 {
		d := acc[i] - m.mean[i]
		m.mean[i] += k * d
		d2 += d * d
	}
	m.v += k * (d2 - m.v)
}

// linear is the variance above the sensor's white noise (density noise,
// m/s²/√Hz, over samples of dt): 3·noise²/dt on the three axes.
func (m *motionPower) linear(noise, dt float32) float32 {
	if !(dt > 0) {
		return 0
	}
	return max(m.v-3*noise*noise/dt, 0)
}
