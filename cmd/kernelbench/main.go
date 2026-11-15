// Command kernelbench times the EasyRobot kernels the tufty2040 badge demo
// runs per IMU sample or per frame, on whichever machine it is built for:
// the badge (tinygo -target=tufty2040, RP2040 Cortex-M0+ at 125 MHz, no FPU
// and no divide instruction) or the host. Per kernel it prints µs per
// call, the heap allocated during the timed loop (must be 0) and an FNV of
// the kernel's outputs, so that an optimisation can be checked for
// bit-identical results on the chip as well as the host. See
// README.md.
package main

import (
	"math"
	"runtime"
	"time"

	"github.com/itohio/EasyRobot/x/math/filter/ahrs"
	"github.com/itohio/EasyRobot/x/math/fluid/flip"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/nav/calib"
	"github.com/itohio/EasyRobot/x/math/nav/eskf"
	"github.com/itohio/EasyRobot/x/math/nav/pdr"
	"github.com/itohio/EasyRobot/x/math/nav/strapdown"
	"github.com/itohio/EasyRobot/x/math/swarm/boids"
)

// samples is one second of a badge in the hand at 100 Hz: resting, then
// tilting and pushed back and forth, from a fixed seed (the same on every
// machine). Accelerometer m/s², gyro rad/s.
const samples = 100

var acc, gyro [samples][3]float32
var quat [samples][4]float32 // Mahony's attitude for each sample

func init() {
	x := uint32(2463534242)
	noise := func(s float32) float32 {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		return s * (float32(x>>8)/(1<<24) - 0.5)
	}
	for i := range samples {
		t := float32(i) / samples
		tilt := 0.4 * t // rad about x
		push := float32(0)
		if i >= samples/2 {
			push = 2 * float32(math.Sin(float64(2*math.Pi*2*t))) // m/s² along y
		}
		acc[i] = [3]float32{noise(0.06), 9.81*float32(math.Sin(float64(tilt))) + push + noise(0.06), 9.81*float32(math.Cos(float64(tilt))) + noise(0.06)}
		gyro[i] = [3]float32{0.4 + noise(0.004), noise(0.004), 0.05 + noise(0.004)}
	}
	m := ahrs.NewMahony(ahrs.WithKP(1), ahrs.WithKI(0), ahrs.WithMagnetometer(false))
	var in mat.Matrix3x3
	for i := range samples {
		in = mat.Matrix3x3{acc[i], gyro[i], {}}
		m.Update(0.01, &in)
		quat[i] = [4]float32(m.Output())
	}
}

// fnv folds 32-bit words into an FNV-1a hash.
type fnv uint32

func newFNV() fnv { return 2166136261 }

func (h *fnv) word(w uint32) {
	for range 4 {
		*h = (*h ^ fnv(w&0xff)) * 16777619
		w >>= 8
	}
}

func (h *fnv) f32(v ...float32) {
	for _, f := range v {
		h.word(math.Float32bits(f))
	}
}

var ms runtime.MemStats

func heap() uint64 {
	runtime.ReadMemStats(&ms)
	return ms.TotalAlloc
}

var start = time.Now()

func now() int64 { return int64(time.Since(start) / time.Microsecond) }

func report(name string, calls int, us int64, alloc uint64, h fnv) {
	println(name, ":", us*1000/int64(calls), "ns/call over", calls, "calls, heap", alloc, "B, fnv", uint32(h))
}

func benchMahony(calls int) {
	m := ahrs.NewMahony(ahrs.WithKP(1), ahrs.WithKI(0), ahrs.WithMagnetometer(false))
	var in mat.Matrix3x3
	a0, t0 := heap(), now()
	for c := range calls {
		i := c % samples
		in = mat.Matrix3x3{acc[i], gyro[i], {}}
		m.Update(0.01, &in)
	}
	t1, a1 := now(), heap()
	h := newFNV()
	q := m.Output()
	h.f32(q[0], q[1], q[2], q[3])
	report("ahrs.Mahony.Update(6-axis)", calls, t1-t0, a1-a0, h)
}

func benchStrapdown(calls int) {
	n, err := strapdown.New(strapdown.DefaultConfig())
	if err != nil {
		println(err.Error())
		return
	}
	a0, t0 := heap(), now()
	for c := range calls {
		i := c % samples
		n.Update(0.01, quat[i], acc[i], gyro[i])
	}
	t1, a1 := now(), heap()
	h := newFNV()
	h.f32(n.P[0], n.P[1], n.P[2], n.V[0], n.V[1], n.V[2])
	report("strapdown.Nav.Update", calls, t1-t0, a1-a0, h)
}

// still is a badge lying still (calibrated, upright), from a fixed seed:
// the filter's at-rest path (zero velocity, zero rate) runs on it.
var stillAcc, stillGyro [samples][3]float32

func init() {
	x := uint32(88172645)
	noise := func(s float32) float32 {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		return s * (float32(x>>8)/(1<<24) - 0.5)
	}
	for i := range samples {
		stillAcc[i] = [3]float32{noise(0.06), 9.81 + noise(0.06), noise(0.06)}
		stillGyro[i] = [3]float32{noise(0.004), noise(0.004), noise(0.004)}
	}
}

// benchESKF times eskf.Filter.Update after its first rest, on the moving
// stream or the still one.
func benchESKF(calls int, moving bool) {
	f, err := eskf.New(eskf.DefaultConfig())
	if err != nil {
		println(err.Error())
		return
	}
	c := eskf.DefaultConfig()
	if err := f.SetRange(c.AccelRange, c.GyroRange); err != nil { // the in-place setter, in the allocation audit
		println(err.Error())
		return
	}
	for c := range 200 { // the first rest: ready
		f.Update(0.01, stillAcc[c%samples], stillGyro[c%samples])
	}
	a, g, name := &stillAcc, &stillGyro, "eskf.Filter.Update(at rest)"
	if moving {
		a, g, name = &acc, &gyro, "eskf.Filter.Update(moving)"
	}
	a0, t0 := heap(), now()
	for c := range calls {
		i := c % samples
		f.Update(0.01, a[i], g[i])
	}
	t1, a1 := now(), heap()
	h := newFNV()
	h.f32(f.Q[0], f.Q[1], f.Q[2], f.Q[3], f.V[0], f.V[1], f.V[2], f.P[0], f.P[1], f.P[2])
	report(name, calls, t1-t0, a1-a0, h)
}

func benchCalibrator(calls int) {
	c, err := calib.New(calib.DefaultConfig())
	if err != nil {
		println(err.Error())
		return
	}
	a0, t0 := heap(), now()
	for k := range calls {
		i := k % samples
		c.Update(0.01, acc[i], gyro[i])
	}
	t1, a1 := now(), heap()
	r, _ := c.Result()
	h := newFNV()
	h.f32(r.GyroBias[0], r.GyroBias[1], r.GyroBias[2])
	report("calib.Calibrator.Update", calls, t1-t0, a1-a0, h)
}

func benchDetector(calls int) {
	d, err := pdr.NewDetector(pdr.DefaultDetectorConfig(100))
	if err != nil {
		println(err.Error())
		return
	}
	steps := uint32(0)
	a0, t0 := heap(), now()
	for c := range calls {
		if d.Update(acc[c%samples]) {
			steps++
		}
	}
	t1, a1 := now(), heap()
	h := newFNV()
	h.word(steps)
	report("pdr.Detector.Update", calls, t1-t0, a1-a0, h)
}

// tilt: gravity 0.02 cell/step² at −40°, 0°, +40° (Q16), as the badge demo.
var tilt = [3][2]int32{{-843, 1004}, {0, 1311}, {843, 1004}}

func benchFlip(steps int) {
	const w, hgt, n = 24, 18, 400 // the demo's Water
	l, err := flip.New(w, hgt, n, flip.DefaultParams())
	if err != nil {
		println(err.Error())
		return
	}
	ras := make([]uint8, 4*w*hgt)
	for s := range 120 { // settle from Reset's packing first, untimed
		g := tilt[(s/120)%3]
		l.GX, l.GY = g[0], g[1]
		l.Step()
	}
	var stepUs, rasUs int64
	a0 := heap()
	for s := 120; s < 120+steps; s++ {
		g := tilt[(s/120)%3]
		l.GX, l.GY = g[0], g[1]
		t0 := now()
		l.Step()
		t1 := now()
		l.Raster(ras, 2*w, 2*hgt)
		t2 := now()
		stepUs += t1 - t0
		rasUs += t2 - t1
	}
	a1 := heap()
	h := newFNV()
	for i := range l.X {
		h.word(uint32(l.X[i]) | uint32(l.Y[i])<<16)
	}
	report("flip.Liquid.Step(24x18/400)", steps, stepUs, a1-a0, h)
	report("flip.Liquid.Raster(48x36)", steps, rasUs, 0, h)
}

func benchBoids(steps int) {
	f, err := boids.New(64, 64, 48, boids.DefaultParams())
	if err != nil {
		println(err.Error())
		return
	}
	f.Box.Spin = true
	a0, t0 := heap(), now()
	for range steps {
		f.Box.Advance()
		f.Step()
	}
	t1, a1 := now(), heap()
	h := newFNV()
	for i := range f.N {
		h.word(uint32(f.X[i]))
		h.word(uint32(f.Y[i]))
	}
	report("boids.Flock.Step(64)", steps, t1-t0, a1-a0, h)
}

func main() {
	for pass := 0; passes == 0 || pass < passes; pass++ {
		println("kernelbench pass", pass)
		benchMahony(1000)
		benchStrapdown(1000)
		benchESKF(1000, true)
		benchESKF(1000, false)
		benchCalibrator(1000)
		benchDetector(1000)
		benchFlip(300)
		benchBoids(300)
	}
}
