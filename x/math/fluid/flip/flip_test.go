package flip

import (
	"fmt"
	"math"
	"testing"
)

// The box and fill the badge uses: 32×24 cells, 700 particles (a pool
// about 7 cells deep), gravity 0.02 cell per step².
const (
	tw, th, tn = 32, 24, 700
	tg         = 0.02
)

func newTest(t testing.TB) *Liquid {
	t.Helper()
	l, err := New(tw, th, tn, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// setG points gravity deg degrees from straight down (+y), counter-
// clockwise as seen on screen (positive turns it towards +x).
func setG(l *Liquid, deg float64) { setGAt(l, deg, tg) }

// setGAt is setG with gravity g cell per step².
func setGAt(l *Liquid, deg, g float64) {
	a := deg * math.Pi / 180
	l.GX = int32(math.Round(g * math.Sin(a) * One))
	l.GY = int32(math.Round(g * math.Cos(a) * One))
}

func steps(l *Liquid, n int) {
	for range n {
		l.Step()
	}
}

// turn rotates gravity from a to b degrees over n steps, stepping each.
func turn(l *Liquid, a, b float64, n int) { turnAt(l, a, b, n, tg) }

// turnAt is turn with gravity g cell per step².
func turnAt(l *Liquid, a, b float64, n int, g float64) {
	for k := 1; k <= n; k++ {
		setGAt(l, a+(b-a)*float64(k)/float64(n), g)
		l.Step()
	}
}

func cells(v int32) float64 { return float64(v) / One }

// pos is a particle coordinate in cells; dpos the difference of two.
func pos(v uint16) float64     { return float64(v) / Cell }
func dpos(a, b uint16) float64 { return float64(int(a)-int(b)) / Cell }

// spd is a particle velocity in cells per step.
func spd(v int16) float64 { return float64(v) / VOne }

// surface returns the particles on the free surface as (s, d): position
// across gravity and depth along it (cells). A particle is on the surface
// when no other particle lies above it (against gravity) within two cells
// and less than 0.6 cell to either side. Particles within 3.5 cells of a
// wall are skipped (the meniscus and the corners).
func surface(l *Liquid) (s, d []float64) {
	gx, gy := cells(l.GX), cells(l.GY)
	gn := math.Hypot(gx, gy)
	gx, gy = gx/gn, gy/gn
	for i := range l.X {
		x, y := pos(l.X[i]), pos(l.Y[i])
		if x < 3.5 || y < 3.5 || x > float64(l.W)-3.5 || y > float64(l.H)-3.5 {
			continue
		}
		top := true
		for j := range l.X {
			dx, dy := pos(l.X[j])-x, pos(l.Y[j])-y
			up, side := -(dx*gx + dy*gy), dx*gy-dy*gx
			if up > 0.05 && up < 2 && math.Abs(side) < 0.6 {
				top = false
				break
			}
		}
		if top {
			s = append(s, x*gy-y*gx)
			d = append(d, x*gx+y*gy)
		}
	}
	return s, d
}

func mean(v []float64) float64 {
	var m float64
	for _, x := range v {
		m += x
	}
	return m / float64(len(v))
}

// level fits a line through the surface particles and returns its tilt
// from perpendicular to gravity (degrees), the depth spread of the
// surface particles (cells) and how many there are.
func level(l *Liquid) (tilt, spread float64, n int) {
	s, d := surface(l)
	ms, md := mean(s), mean(d)
	var sxy, sxx float64
	lo, hi := math.Inf(1), math.Inf(-1)
	for i := range s {
		sxy += (s[i] - ms) * (d[i] - md)
		sxx += (s[i] - ms) * (s[i] - ms)
		lo, hi = min(lo, d[i]), max(hi, d[i])
	}
	return math.Atan(sxy/sxx) * 180 / math.Pi, hi - lo, len(d)
}

// Acceptance: at rest the surface lies perpendicular to gravity, whichever
// way gravity points: a line fitted through the surface particles is
// within 3° of perpendicular, and no surface particle is more than about
// two particle diameters off the others. A wedge of liquid in a corner
// takes the longest to finish creeping into place (a few thousand steps).
func TestRestSurfacePerpendicularToGravity(t *testing.T) {
	for _, deg := range []float64{0, 25, -40, -60, 90, 135, 180} {
		l := newTest(t)
		setG(l, 0)
		steps(l, 100)
		turn(l, 0, deg, 60)
		steps(l, 5000)
		tilt, spread, n := level(l)
		if n < 10 {
			t.Fatalf("%v°: only %d surface particles", deg, n)
		}
		t.Logf("%v°: %d surface particles, tilt %.1f°, depth spread %.2f cells", deg, n, tilt, spread)
		if math.Abs(tilt) > 3 || spread > 1.5 {
			t.Errorf("%v°: surface not perpendicular to gravity: tilt %.1f°, depth spread %.2f cells", deg, tilt, spread)
		}
	}
}

// Acceptance: no jitter when still, whichever way gravity points — at
// every 5° and at five strengths across the tuned range (360 cases), once
// settled (6000 steps after the turn), no particle drifts more than a
// twentieth of a cell in 1000 steps, and drawn at two pixels per cell
// after every step, at most ten pixel changes (lit ↔ dark) occur in all
// 1000 frames. Lone surface particles rocking across a cell edge once kept
// a few directions flickering for ever (60°: 130 changes per 1000 steps;
// without sticky cells 95°, at 0.0125 only: 472). -short checks every 30°
// at one strength.
func TestStillLiquidDoesNotJitter(t *testing.T) {
	every, gs := 5, []float64{0.01, 0.0125, 0.015, 0.0175, 0.02}
	if testing.Short() {
		every, gs = 30, []float64{tg}
	}
	for _, g := range gs {
		for deg := 0; deg < 360; deg += every {
			t.Run(fmt.Sprintf("%d°/g%v", deg, g), func(t *testing.T) {
				t.Parallel()
				l := newTest(t)
				setGAt(l, 0, g)
				steps(l, 100)
				turnAt(l, 0, float64(deg), 60, g)
				steps(l, 6000)
				x0, y0 := append([]uint16(nil), l.X...), append([]uint16(nil), l.Y...)
				pw, ph := 2*tw, 2*th
				prev, cur := make([]uint8, pw*ph), make([]uint8, pw*ph)
				l.Raster(prev, pw, ph)
				flicker := 0
				for range 1000 {
					l.Step()
					l.Raster(cur, pw, ph)
					for k := range cur {
						if (prev[k] == 0) != (cur[k] == 0) {
							flicker++
						}
					}
					prev, cur = cur, prev
				}
				var moved float64
				for i := range l.X {
					moved = max(moved, math.Hypot(dpos(l.X[i], x0[i]), dpos(l.Y[i], y0[i])))
				}
				if moved > 0.05 || flicker > 10 {
					t.Errorf("liquid at rest moves: max drift %.4f cells, %d pixel changes in 1000 frames", moved, flicker)
				}
			})
		}
	}
}

func meanX(l *Liquid) float64 {
	var s float64
	for _, x := range l.X {
		s += pos(x)
	}
	return s / float64(len(l.X))
}

// Acceptance: momentum — after a tilt and back the liquid sloshes past
// its rest position several times, losing a little each time, and then
// settles: still, with its surface level (as in
// TestRestSurfacePerpendicularToGravity, within 3.5°). Rows of particles
// lie 0.57 cell apart; the last part-filled row comes to rest lying to
// one side, which a line fit reads as a tilt of up to about 3°, and which
// moves the mean x up to about 0.6 cell from where it started.
func TestSloshesAndSettlesAfterTilt(t *testing.T) {
	for _, c := range []struct{ deg, g float64 }{{30, tg}, {-30, 0.01}, {45, 0.015}, {20, 0.015}} {
		t.Run(fmt.Sprintf("%v°/g%v", c.deg, c.g), func(t *testing.T) {
			t.Parallel()
			l := newTest(t)
			setGAt(l, 0, c.g)
			steps(l, 300)
			rest := meanX(l)
			turnAt(l, 0, c.deg, 10, c.g)
			steps(l, 150)
			turnAt(l, c.deg, 0, 10, c.g)
			var peaks []float64 // |excursion| between successive crossings of rest
			side, peak := 0, 0.0
			for range 1500 {
				l.Step()
				e := meanX(l) - rest
				s := 1
				if e < 0 {
					s = -1
				}
				if math.Abs(e) > 0.05 && s != side {
					if side != 0 {
						peaks = append(peaks, peak)
					}
					side, peak = s, 0
				}
				peak = max(peak, math.Abs(e))
			}
			t.Logf("slosh peaks (cells of mean-x excursion): %.2f", peaks)
			if len(peaks) < 3 {
				t.Fatalf("liquid crossed its rest position only %d times: no momentum", len(peaks))
			}
			if peaks[len(peaks)-1] >= peaks[0] {
				t.Errorf("sloshing does not decay: first %.2f, last %.2f", peaks[0], peaks[len(peaks)-1])
			}
			steps(l, 1500)
			x0, y0 := append([]uint16(nil), l.X...), append([]uint16(nil), l.Y...)
			steps(l, 100)
			var moved float64
			for i := range l.X {
				moved = max(moved, math.Hypot(dpos(l.X[i], x0[i]), dpos(l.Y[i], y0[i])))
			}
			tilt, spread, _ := level(l)
			e := math.Abs(meanX(l) - rest)
			t.Logf("settled: tilt %.1f°, depth spread %.2f cells, mean x %.2f cells from rest", tilt, spread, e)
			if moved > 0.05 || math.Abs(tilt) > 3.5 || spread > 1.5 || e > 0.75 {
				t.Errorf("not settled after 3000 steps: a particle moved %.3f cells in 100 steps, tilt %.1f°, spread %.2f, mean x %.2f cells off", moved, tilt, spread, e)
			}
		})
	}
}

// Acceptance: a slight tilt moves the liquid — a pool settled under
// gravity 1° or 2° to one side, turned to as much to the other, moves its
// mean x that way by at least 0.3 cell (a level surface would move 0.37 or
// 0.75; the last part-filled row of particles tends to slide all the way).
// Starting from one side rather than straight down keeps Reset's
// part-filled top row, which lies at the left, from deciding the result.
// A still particle damps a push back against its last move, which a turn
// back the other way is, until gravity has turned far enough for the last
// moves to be forgotten; without that, or with every push damped, these
// tilts do nothing.
func TestSlightTiltMovesTheLiquid(t *testing.T) {
	for _, g := range []float64{0.01, tg} {
		for _, deg := range []float64{2, -2, 1, -1} {
			t.Run(fmt.Sprintf("%v°/g%v", deg, g), func(t *testing.T) {
				t.Parallel()
				l := newTest(t)
				setGAt(l, 0, g)
				steps(l, 100)
				turnAt(l, 0, -deg, 60, g)
				steps(l, 4000)
				m0 := meanX(l)
				turnAt(l, -deg, deg, 60, g)
				steps(l, 2000)
				shift := (meanX(l) - m0) * math.Copysign(1, deg)
				t.Logf("mean x moved %.2f cells with the tilt", shift)
				if shift < 0.3 {
					t.Errorf("turning gravity from %v° to %v° moved the mean x only %.2f cells its way", -deg, deg, shift)
				}
			})
		}
	}
}

// Physics: the slosh period is the first mode of a standing wave in a
// pool of width L and depth h, T = 2π/√(g·k·tanh(k·h)) with k = π/L. The
// particles slosh a little slower than linear theory (a 30° kick is not a
// small wave, and the grid adds drag): within 15 %. Halving gravity has to
// lengthen the period by √2, within 5 %.
func TestSloshPeriodMatchesWaveTheory(t *testing.T) {
	gs := []float64{0.01, tg}
	var periods [2]float64
	for n, g := range gs {
		l := newTest(t)
		setGAt(l, 0, g)
		steps(l, 2000)
		rest, h := meanX(l), columnHeight(l, tw/2)
		turnAt(l, 0, 30, 10, g)
		steps(l, 150)
		turnAt(l, 30, 0, 10, g)
		var flips []int // steps at which the excursion changes side
		side := 0
		for k := range 1500 {
			l.Step()
			e := meanX(l) - rest
			if math.Abs(e) < 0.05 {
				continue
			}
			s := int(math.Copysign(1, e))
			if s != side {
				if side != 0 {
					flips = append(flips, k)
				}
				side = s
			}
		}
		if len(flips) < 5 {
			t.Fatalf("g %v: only %d half-swings", g, len(flips))
		}
		periods[n] = 2 * float64(flips[4]-flips[0]) / 4
		kw := math.Pi / float64(tw-2)
		want := 2 * math.Pi / math.Sqrt(g*kw*math.Tanh(kw*h))
		t.Logf("g %v, depth %.2f cells: period %.0f steps, theory %.0f", g, h, periods[n], want)
		if math.Abs(periods[n]-want) > 0.15*want {
			t.Errorf("g %v: slosh period %.0f steps, linear wave theory %.0f", g, periods[n], want)
		}
	}
	if r := periods[0] / periods[1]; math.Abs(r-math.Sqrt2) > 0.05*math.Sqrt2 {
		t.Errorf("half the gravity lengthens the period %.3f times, want √2", r)
	}
}

// columnHeight is how high the liquid stands at column x (cells above the
// floor), from the topmost particle within half a cell of x.
func columnHeight(l *Liquid, x float64) float64 {
	top := float64(l.H - 1)
	for i := range l.X {
		if math.Abs(pos(l.X[i])-x) < 0.5 {
			top = min(top, pos(l.Y[i]))
		}
	}
	return float64(l.H-1) - top
}

// Acceptance: waves travel and reflect — a dam of liquid released at the
// left runs to the right wall no faster than the ideal frictionless front
// speed 2·√(g·h₀) (Ritter) and not far below it, climbs the wall, and the
// reflected wave comes back and raises the level at the left wall again.
func TestDamBreakWaveTravelsAndReflects(t *testing.T) {
	l, err := New(tw, th, 220, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	// a block 8 cells wide at the left wall, hexagonally packed
	r := DefaultParams().Radius >> 8
	x, y, row := int32(Cell)+r, int32(th-1)*Cell-r, 0
	right := int32(9) * Cell
	for i := range l.X {
		l.X[i], l.Y[i], l.VX[i], l.VY[i] = uint16(x), uint16(y), 0, 0
		x += 2 * r
		if x > right-r {
			row++
			x = int32(Cell) + r + int32(row&1)*r
			y -= 2 * r * 87 / 100
		}
	}
	h0 := columnHeight(l, 5)
	setG(l, 0)
	leftX, rightX := 1.8, float64(tw)-1.8
	arrive, ret := -1, -1
	minLeft := h0
	for s := range 600 {
		l.Step()
		hl, hr := columnHeight(l, leftX), columnHeight(l, rightX)
		if arrive < 0 && hr > 1 {
			arrive = s
		}
		if arrive >= 0 && ret < 0 {
			minLeft = min(minLeft, hl)
			if hl > minLeft+1.5 {
				ret = s
			}
		}
	}
	dist := rightX - 9
	ritter := dist / (2 * math.Sqrt(tg*h0))
	t.Logf("dam %.1f cells high: front reached the right wall at step %d (Ritter %.0f), reflection back at the left wall at step %d", h0, arrive, ritter, ret)
	if arrive < 0 || float64(arrive) < ritter || float64(arrive) > 2.5*ritter {
		t.Errorf("front arrival at step %d, want between %.0f (Ritter) and %.0f", arrive, ritter, 2.5*ritter)
	}
	if ret < 0 || ret <= arrive {
		t.Errorf("no reflected wave back at the left wall (arrive %d, return %d)", arrive, ret)
	}
}

// isolated counts particles with no other particle within one cell —
// droplets in flight.
func isolated(l *Liquid) int {
	n := 0
	for i := range l.X {
		alone := true
		for j := range l.X {
			if i == j {
				continue
			}
			dx, dy := dpos(l.X[i], l.X[j]), dpos(l.Y[i], l.Y[j])
			if dx*dx+dy*dy < 1 {
				alone = false
				break
			}
		}
		if alone {
			n++
		}
	}
	return n
}

// Acceptance: a fast tilt splashes — droplets leave the body of liquid —
// where a slow tilt to the same angle does not, and the still liquid has
// none.
func TestFastTiltSplashes(t *testing.T) {
	run := func(over int) (atRest, most int) {
		l := newTest(t)
		setG(l, 0)
		steps(l, 300)
		atRest = isolated(l)
		for k := 1; k <= over; k++ {
			setG(l, 100*float64(k)/float64(over))
			l.Step()
			if k%4 == 0 {
				most = max(most, isolated(l))
			}
		}
		for k := range 200 {
			l.Step()
			if k%4 == 0 {
				most = max(most, isolated(l))
			}
		}
		return atRest, most
	}
	rest, fast := run(4)
	_, slow := run(400)
	t.Logf("droplets: at rest %d, fast tilt %d, slow tilt %d", rest, fast, slow)
	if rest != 0 {
		t.Errorf("%d droplets in a still liquid", rest)
	}
	if fast < 5 || fast <= 2*slow {
		t.Errorf("a fast tilt gives %d droplets, a slow one %d: want a splash", fast, slow)
	}
}

// Acceptance: volume conserved — the particle count never changes, and
// after being thrown around the liquid settles back to within half a cell
// (about 6 % of the 8-cell pool) of the level it started from; the rest is
// how the particles happen to pack.
func TestVolumeConservedThroughShaking(t *testing.T) {
	l := newTest(t)
	setG(l, 0)
	steps(l, 400)
	_, d := surface(l)
	before := mean(d)
	for _, a := range []float64{60, -60, 120, -30, 0} {
		turn(l, 0, a, 5)
		steps(l, 80)
		turn(l, a, 0, 5)
	}
	steps(l, 2000)
	_, d = surface(l)
	after := mean(d)
	t.Logf("surface %.2f cells below the ceiling before, %.2f after", before, after)
	if math.Abs(after-before) > 0.5 {
		t.Errorf("level changed by %.2f cells: volume not conserved", after-before)
	}
}

// Two overlapping particles in weightless liquid end one step later two
// radii apart (each pushed by half the overlap; the second pass finds
// nothing left), on the line they started on.
func TestOverlappingParticlesEndTwoRadiiApart(t *testing.T) {
	l, err := New(8, 8, 2, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	l.X[0], l.Y[0] = 4*Cell, 4*Cell
	l.X[1], l.Y[1] = 4*Cell+Cell/5, 4*Cell+Cell/10 // 0.22 cell away
	l.Step()
	dx, dy := dpos(l.X[1], l.X[0]), dpos(l.Y[1], l.Y[0])
	d0 := 2 * float64(DefaultParams().Radius>>8) / Cell // the radius as positions hold it
	if d := math.Hypot(dx, dy); math.Abs(d-d0) > 0.01 || math.Abs(dy/dx-0.5) > 0.02 {
		t.Errorf("after one step: %.3f cells apart along slope %.3f, want %.3f along 0.5", d, dy/dx, d0)
	}
}

// A particle thrown at a wall stops at it (a radius off the ring) rather
// than keeping its speed into the wall.
func TestParticleStopsAtWall(t *testing.T) {
	l, err := New(8, 8, 1, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	r := DefaultParams().Radius >> 8
	l.X[0], l.Y[0], l.VX[0] = uint16(7*Cell-r-Cell/10), 4*Cell, VOne/2
	l.Step()
	if l.X[0] != uint16(7*Cell-r) || l.VX[0] > 0 {
		t.Errorf("x %.3f (wall at %.3f), vx %.3f: want at the wall, not moving into it", pos(l.X[0]), float64(7*Cell-r)/Cell, spd(l.VX[0]))
	}
}

// Reset after the liquid has been thrown around is as good as a new one:
// nothing from before (pressure, liquid cells, last moves) carries over,
// so the two step identically — also with a particle placed a hair under
// a cell that was liquid before the Reset (it would keep that cell
// liquid), and under the same gravity as before the Reset (nothing then
// tells the liquid to forget its last moves).
func TestResetForgetsThePast(t *testing.T) {
	l, fresh := newTest(t), newTest(t)
	setG(l, 0)
	turn(l, 0, 180, 10) // liquid against the ceiling: the top cells are liquid
	steps(l, 300)
	l.Reset()
	for _, m := range []*Liquid{l, fresh} {
		setG(m, 0)
		m.X[0], m.Y[0] = 10*Cell+Cell/2, 5*Cell+Cell/50
	}
	steps(l, 50)
	steps(fresh, 50)
	same(t, "after Reset from the ceiling", l, fresh)

	l, fresh = newTest(t), newTest(t)
	setG(l, 30)
	steps(l, 2000) // settled: particles stand with their last moves known
	l.Reset()
	setG(fresh, 30)
	steps(l, 400)
	steps(fresh, 400)
	same(t, "after Reset under unchanged gravity", l, fresh)
}

// same fails unless a and b have identical particles.
func same(t *testing.T, what string, a, b *Liquid) {
	t.Helper()
	for i := range a.X {
		if a.X[i] != b.X[i] || a.Y[i] != b.Y[i] || a.VX[i] != b.VX[i] || a.VY[i] != b.VY[i] {
			t.Fatalf("%s: particle %d differs from a new liquid's", what, i)
		}
	}
}

// However fast the caller throws a particle, it moves at most MaxV per
// step and keeps at most that speed.
func TestSpeedLimit(t *testing.T) {
	l, err := New(16, 16, 1, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	m := DefaultParams().MaxV
	l.X[0], l.Y[0], l.VX[0], l.VY[0] = 8*Cell, 8*Cell, 32767, -32767 // 2 cells per step
	x0, y0 := l.X[0], l.Y[0]
	l.Step()
	if dx, dy := dpos(l.X[0], x0), dpos(l.Y[0], y0); dx > cells(m)+0.5/Cell || -dy > cells(m)+0.5/Cell || spd(l.VX[0]) > cells(m) || -spd(l.VY[0]) > cells(m) {
		t.Errorf("moved (%.3f, %.3f) at (%.3f, %.3f) per step; limit %.3f", dx, dy, spd(l.VX[0]), spd(l.VY[0]), cells(m))
	}
}

// Size is what New allocates: the bytes of every slice it makes, for the
// badge sizes and a tall box.
func TestSizeIsWhatNewAllocates(t *testing.T) {
	for _, c := range [][3]int{{32, 24, 700}, {28, 21, 524}, {16, 12, 148}, {10, 40, 300}} {
		l, err := New(c[0], c[1], c[2], DefaultParams())
		if err != nil {
			t.Fatal(err)
		}
		got := 2*(len(l.X)+len(l.Y)+len(l.VX)+len(l.VY)+len(l.order)) + len(l.last) +
			4*(len(l.u)+len(l.v)) + 2*(len(l.pu)+len(l.pv)) + len(l.cell) + 2*(len(l.aux)+len(l.pres))
		if want := Size(c[0], c[1], c[2]); got != want {
			t.Errorf("%dx%d, %d particles: New allocates %d B, Size says %d", c[0], c[1], c[2], got, want)
		}
	}
}

// A deep pool (16×64 cells, about 50 deep) stays still: its pressure
// passes what 16 bits hold (0.02 cell per step² under 50 cells), and
// saturates instead of wrapping round.
func TestDeepPoolStaysStill(t *testing.T) {
	l, err := New(16, 64, 1850, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	setG(l, 0)
	steps(l, 3000)
	x0, y0 := append([]uint16(nil), l.X...), append([]uint16(nil), l.Y...)
	steps(l, 300)
	var moved float64
	for i := range l.X {
		moved = max(moved, math.Hypot(dpos(l.X[i], x0[i]), dpos(l.Y[i], y0[i])))
	}
	if top := columnHeight(l, 8); moved > 0.05 || top < 40 {
		t.Errorf("deep pool: a particle moved %.3f cells in 300 steps, depth %.1f cells", moved, top)
	}
}

func TestStepDoesNotAllocate(t *testing.T) {
	l := newTest(t)
	setG(l, 20)
	if a := testing.AllocsPerRun(50, l.Step); a != 0 {
		t.Errorf("Step allocates %v times", a)
	}
	ras := make([]uint8, 64*48)
	if a := testing.AllocsPerRun(50, func() { l.Raster(ras, 64, 48) }); a != 0 {
		t.Errorf("Raster allocates %v times", a)
	}
}

func TestRasterMapsCellsToPixels(t *testing.T) {
	l, err := New(4, 4, 3, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	l.X[0], l.Y[0] = Cell+Cell/2, Cell+Cell/2   // (1.5, 1.5)
	l.X[1], l.Y[1] = Cell+Cell/2, Cell+Cell/2+1 // same pixel
	l.X[2], l.Y[2] = 3*Cell-1, 2*Cell           // just left of x = 3, on y = 2
	ras := make([]uint8, 12*8)                  // 3 px per cell across, 2 down
	l.Raster(ras, 12, 8)
	if ras[3*12+4] != 2 || ras[4*12+8] != 1 {
		t.Errorf("raster wrong: (4,3)=%d want 2, (8,4)=%d want 1", ras[3*12+4], ras[4*12+8])
	}
}

func TestNewRejectsParams(t *testing.T) {
	ok := DefaultParams()
	bad := func(f func(*Params)) Params { p := ok; f(&p); return p }
	cases := []struct {
		name    string
		w, h, n int
		p       Params
	}{
		{"narrow", 2, 10, 10, ok},
		{"huge", 256, 10, 10, ok},
		{"too many particles", 10, 10, 65536, ok},
		{"empty", 10, 10, 0, ok},
		{"radius too small", 10, 10, 10, bad(func(p *Params) { p.Radius = One/8 - 1 })},
		{"radius half a cell", 10, 10, 10, bad(func(p *Params) { p.Radius = One / 2 })},
		{"speed over a cell", 10, 10, 10, bad(func(p *Params) { p.MaxV = One + 1 })},
		{"flip over 1", 10, 10, 10, bad(func(p *Params) { p.FlipQ8 = 257 })},
		{"over-relaxation over 2", 10, 10, 10, bad(func(p *Params) { p.OverQ8 = 513 })},
		{"calm too small", 10, 10, 10, bad(func(p *Params) { p.Calm = One/64 - 1 })},
		{"calm over a cell", 10, 10, 10, bad(func(p *Params) { p.Calm = One + 1 })},
		{"rest over 4 g", 10, 10, 10, bad(func(p *Params) { p.RestQ8 = 1025 })},
		{"hold over 1", 10, 10, 10, bad(func(p *Params) { p.HoldQ8 = 257 })},
		{"negative hold", 10, 10, 10, bad(func(p *Params) { p.HoldQ8 = -1 })},
		{"stick over half a cell", 10, 10, 10, bad(func(p *Params) { p.Stick = One/2 + 1 })},
		{"negative drift", 10, 10, 10, bad(func(p *Params) { p.DriftQ8 = -1 })},
	}
	for _, c := range cases {
		if _, err := New(c.w, c.h, c.n, c.p); err != ErrParams {
			t.Errorf("%s: err %v, want ErrParams", c.name, err)
		}
	}
	if _, err := New(3, 3, 1, ok); err != nil {
		t.Errorf("smallest box: %v", err)
	}
}

func BenchmarkStep(b *testing.B) {
	l := newTest(b)
	setG(l, 0)
	steps(l, 300)
	turn(l, 0, 30, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		l.Step()
	}
}
