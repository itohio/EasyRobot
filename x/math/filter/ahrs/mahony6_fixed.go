//go:build fixedmotion

package ahrs

import (
	"math"
	"math/bits"
)

// fixedPath reports whether the 6-axis update runs in fixed point. With
// -tags fixedmotion it does: for cores without an FPU, where every float32
// operation is a software routine (the RP2040's Cortex-M0+), integer
// arithmetic is several times cheaper. It is never selected automatically;
// the float32 path is the default and the reference. The attitude follows
// the float32 path within 1e-4 rad (TestFixedMahonyFollowsFloat).
const fixedPath = true

// fixed6 is the 6-axis filter's state in fixed point: the attitude in Q30
// (2³⁰ = 1) and the integral of the error in Q30, summed. The gains are
// kept in Q28 (0.005 to 1e-6 relative: in Q16 its 0.1 % error, carried by a
// large integral, moved the attitude 2.5e-4 rad), converted again only when
// the float gains change.
type fixed6 struct {
	q        [4]int32  // Q30
	eInt     [3]int64  // Q30
	wq       [4]uint32 // the bits last written to m.q,
	we       [3]uint32 // and to m.eInt: anything else there was set from outside
	kpF, kiF float32   // the gains kp and ki were made from
	kp, ki   int64     // Q28
	dtF      float32   // the sample period hdt was made from
	hdt      int64     // dt/2 in Q30
}

// rsh is x / 2^s rounded to nearest (halves up), so that repeated
// rescaling carries no bias.
func rsh(x int64, s uint) int64 { return (x + 1<<(s-1)) >> s }

// mulQ is a·b / 2^s rounded to nearest, through a 128-bit product: the
// integral can grow past what an int64 product with the gain holds.
func mulQ(a, b int64, s uint) int64 {
	neg := (a < 0) != (b < 0)
	ua, ub := uint64(a), uint64(b)
	if a < 0 {
		ua = uint64(-a)
	}
	if b < 0 {
		ub = uint64(-b)
	}
	hi, lo := bits.Mul64(ua, ub)
	lo, c := bits.Add64(lo, 1<<(s-1), 0)
	hi += c
	r := int64(hi<<(64-s) | lo>>s)
	if neg {
		return -r
	}
	return r
}

// isqrt is ⌊√x⌋ for x ≥ 0 (digit by digit, no divide).
func isqrt(x int64) int64 {
	var r int64
	b := int64(1) << 62
	for b > x {
		b >>= 2
	}
	for b != 0 {
		if x >= r+b {
			x -= r + b
			r = r>>1 + b
		} else {
			r >>= 1
		}
		b >>= 2
	}
	return r
}

// fix is x·2^frac rounded to nearest (halves away from zero), read from
// the float's bits with integer shifts: no float operation, and no bias
// (a truncating conversion pulls every value towards zero by half a step,
// which near the filter's upside-down start decides where it turns).
// |x|·2^frac must stay under 2⁶²; NaN and ±Inf are not expected.
func fix(x float32, frac uint) int64 {
	b := math.Float32bits(x)
	e := int(b>>23&0xff) - 150 + int(frac) // x = m·2^(exp−150) for the 24-bit m
	m := int64(b&0x7fffff | 0x800000)
	if b>>23&0xff == 0 { // zero or subnormal
		m = int64(b & 0x7fffff)
		e++
	}
	switch {
	case e >= 0:
		m <<= uint(e)
	case e > -40:
		m = (m + 1<<uint(-e-1)) >> uint(-e)
	default:
		m = 0
	}
	if b>>31 != 0 {
		return -m
	}
	return m
}

// update6 is calculateWOMag in fixed point: accelerometer in Q22 m/s² (±512),
// gyro in Q24 rad/s (±128 rad/s), unit vectors and the attitude in Q30,
// products in int64. It reads the same inputs and leaves the attitude in
// m.q as float32.
func (m *MahonyAHRS) update6() {
	f := &m.fx
	// the float state is the filter's state: take it up whenever it is not
	// what this path last wrote (a new filter, Reset, a state set directly)
	if f.changed(m) {
		for i := range 4 {
			f.q[i] = int32(fix(m.q[i], 30))
		}
		for i := range 3 {
			f.eInt[i] = fix(m.eInt[i], 30)
		}
	}
	if m.GainP != f.kpF || f.kp == 0 && m.GainP != 0 {
		f.kpF, f.kp = m.GainP, fix(m.GainP, 28)
	}
	if m.GainI != f.kiF || f.ki == 0 && m.GainI != 0 {
		f.kiF, f.ki = m.GainI, fix(m.GainI, 28)
	}
	if m.SamplePeriod != f.dtF || f.hdt == 0 {
		f.dtF, f.hdt = m.SamplePeriod, fix(m.SamplePeriod, 29)
	}

	// the measured direction of gravity, a unit vector in Q30
	ax := fix(m.inputMatrix[0][0], 22)
	ay := fix(m.inputMatrix[0][1], 22)
	az := fix(m.inputMatrix[0][2], 22)
	var ux, uy, uz int64
	if n2 := ax*ax + ay*ay + az*az; n2 > 0 {
		n := isqrt(n2) // |a| in Q22 (the squares fit while each axis stays under 724 m/s² and |a| under 418·√3)
		if n >= 1<<22 {
			inv := (int64(1) << 52) / n // 2³⁰/|a|, at most 2³⁰: the products fit
			ux, uy, uz = rsh(ax*inv, 22), rsh(ay*inv, 22), rsh(az*inv, 22)
		} else { // under 1 m/s²: divide each (a·2³⁰ fits)
			ux, uy, uz = (ax<<30)/n, (ay<<30)/n, (az<<30)/n
		}
	}

	// the estimated direction of gravity, Q30
	q1, q2, q3, q4 := int64(f.q[0]), int64(f.q[1]), int64(f.q[2]), int64(f.q[3])
	v0 := rsh(q2*q4-q1*q3, 29)
	v1 := rsh(q1*q2+q3*q4, 29)
	v2 := rsh(q1*q1-q2*q2-q3*q3+q4*q4, 30)

	// the error a × v, Q30
	e0 := rsh(uy*v2-uz*v1, 30)
	e1 := rsh(uz*v0-ux*v2, 30)
	e2 := rsh(ux*v1-uy*v0, 30)
	if f.ki > 0 {
		f.eInt[0] += e0
		f.eInt[1] += e1
		f.eInt[2] += e2
	} else {
		f.eInt = [3]int64{}
	}

	// the corrected rate, Q24 rad/s
	gx := fix(m.inputMatrix[1][0], 24) + mulQ(f.kp, e0, 34)
	gy := fix(m.inputMatrix[1][1], 24) + mulQ(f.kp, e1, 34)
	gz := fix(m.inputMatrix[1][2], 24) + mulQ(f.kp, e2, 34)
	if f.ki > 0 {
		gx += mulQ(f.ki, f.eInt[0], 34)
		gy += mulQ(f.ki, f.eInt[1], 34)
		gz += mulQ(f.ki, f.eInt[2], 34)
	}

	// half the step's rotation, Q30 rad
	hx, hy, hz := rsh(gx*f.hdt, 24), rsh(gy*f.hdt, 24), rsh(gz*f.hdt, 24)

	// integrate the rate of change of the quaternion
	n0 := q1 + rsh(-q2*hx-q3*hy-q4*hz, 30)
	n1 := q2 + rsh(q1*hx+q3*hz-q4*hy, 30)
	n2 := q3 + rsh(q1*hy-q2*hz+q4*hx, 30)
	n3 := q4 + rsh(q1*hz+q2*hy-q3*hx, 30)

	// normalise: |q| stays near 1, so three Newton steps of 1/√s from 1
	// (s = |q|², y ← y(3 − s·y²)/2) reach Q30's resolution
	if s := rsh(n0*n0+n1*n1+n2*n2+n3*n3, 30); s > 0 {
		y := int64(1) << 30
		for range 3 {
			y2 := rsh(y*y, 30)
			y = rsh(y*(3<<30-rsh(s*y2, 30)), 31)
		}
		f.q = [4]int32{int32(rsh(n0*y, 30)), int32(rsh(n1*y, 30)), int32(rsh(n2*y, 30)), int32(rsh(n3*y, 30))}
	}
	for i := range 4 {
		m.q[i] = float32(f.q[i]) * (1.0 / (1 << 30))
	}
	// the integral as the float path keeps it (zero without an integral gain)
	if f.ki > 0 {
		for i := range 3 {
			m.eInt[i] = float32(f.eInt[i]) * (1.0 / (1 << 30))
		}
	} else {
		m.eInt = [3]float32{}
	}
	for i := range 4 {
		f.wq[i] = math.Float32bits(m.q[i])
	}
	for i := range 3 {
		f.we[i] = math.Float32bits(m.eInt[i])
	}
}

// changed reports whether m's float state differs, bit for bit, from what
// update6 last wrote (always for a new or Reset filter: wq is all zero bits,
// which no unit quaternion has).
func (f *fixed6) changed(m *MahonyAHRS) bool {
	for i := range 4 {
		if math.Float32bits(m.q[i]) != f.wq[i] {
			return true
		}
	}
	for i := range 3 {
		if math.Float32bits(m.eInt[i]) != f.we[i] {
			return true
		}
	}
	return false
}
