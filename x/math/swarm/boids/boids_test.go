package boids

import (
	"runtime"
	"testing"
)

// The badge's flock: a 64×48-cell box, the default parameters.
func badge(t testing.TB, n int) *Flock {
	t.Helper()
	f, err := New(n, 64, 48, DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// speed is the octagonal norm the package limits.
func speed(vx, vy int32) int32 {
	ax, ay := max(vx, -vx), max(vy, -vy)
	return max(ax, ay) + min(ax, ay)>>2 + min(ax, ay)>>3
}

// checkInside fails if a boid is outside the box or flies outside the
// speed limits (±3 for the integer rounding of scaled and rotated
// components).
func checkInside(t *testing.T, f *Flock, step int) {
	t.Helper()
	for i := range f.N {
		bx, by := f.Box.ToBox(f.X[i], f.Y[i])
		if bx < -2 || by < -2 || bx > f.Box.W+1 || by > f.Box.H+1 {
			t.Fatalf("step %d boid %d outside the box: %d,%d", step, i, bx, by)
		}
		if s := speed(f.VX[i], f.VY[i]); s > f.VMax+3 || s < f.VMin-3 {
			t.Fatalf("step %d boid %d speed %d outside [%d, %d]", step, i, s, f.VMin, f.VMax)
		}
	}
}

// Over many steps with the largest flock, while the box
// turns, spins and shakes and the flock scatters, every boid stays inside
// the box and flies between the speed limits.
func TestStayInTheBoxWithinSpeed(t *testing.T) {
	f := badge(t, 128)
	f.Box.Spin = true
	for s := range 3000 {
		switch s % 500 {
		case 100:
			f.Box.Target += 3 * TurnStep
		case 250:
			f.Scatter()
		case 300:
			f.Box.Shake(uint8(s), 10*Fix)
		}
		f.Box.Advance()
		f.Step()
		checkInside(t, f, s)
	}
}

// Acceptance (ER-063): 10,000 steps with the box turned 22.5° per press,
// a press every 40 steps in alternating bursts: no boid leaves the box and
// every speed stays within [VMin, VMax].
func TestTurnPerPressFor10000Steps(t *testing.T) {
	f := badge(t, 128)
	for s := range 10000 {
		if s%40 == 0 {
			if s/400%2 == 0 {
				f.Box.Target += TurnStep
			} else {
				f.Box.Target -= TurnStep
			}
		}
		f.Box.Advance()
		f.Step()
		checkInside(t, f, s)
	}
}

// Turning the box doesn't turn the flock. At any box
// angle, a boid no wall reaches keeps its world position and velocity, so
// on screen it appears to turn the other way.
func TestKeepWorldHeadingWhenTheBoxTurns(t *testing.T) {
	f := badge(t, 1)
	for angle := range 256 {
		f.Box.Angle = angle
		f.X[0], f.Y[0] = f.Box.CX+3*Fix, f.Box.CY-2*Fix
		f.VX[0], f.VY[0] = f.VMax, 0
		f.walls(0)
		if f.X[0] != f.Box.CX+3*Fix || f.Y[0] != f.Box.CY-2*Fix || f.VX[0] != f.VMax || f.VY[0] != 0 {
			t.Fatalf("angle %d: a boid in the middle was changed to %d,%d v %d,%d", angle, f.X[0], f.Y[0], f.VX[0], f.VY[0])
		}
	}
	f.Box.Angle = 64 // a quarter turn: east in the world is vertical on screen
	if vx, vy := f.Box.VecToBox(f.VMax, 0); max(vx, -vx) > 1 || max(vy, -vy) < f.VMax-1 {
		t.Fatalf("after a quarter turn the boid should fly vertically on screen, got %d,%d", vx, vy)
	}
}

// A wall that moves pushes the boids it reaches. Shaking
// the box drags its walls through boids resting near one; they end up
// inside, moved along.
func TestPushedByAMovingWall(t *testing.T) {
	f := badge(t, 32)
	for i := range f.N { // all near the left wall, flying up and down
		f.X[i], f.Y[i] = f.Margin+256, int32(i%32)*Fix+Fix
		f.VX[i], f.VY[i] = 0, f.VMin
	}
	f.Box.Shake(0, 10*Fix) // throw the box right (+x)
	x0 := f.X[0]
	for range ShakeLen / 2 {
		f.Box.Advance()
		f.Step()
	}
	for i := range f.N {
		if bx, _ := f.Box.ToBox(f.X[i], f.Y[i]); bx < 0 {
			t.Fatalf("boid %d left behind outside the wall: box x %d", i, bx)
		}
	}
	if f.X[0]-x0 < 10*Fix/2 {
		t.Fatalf("the wall moved %d but pushed the boid only %d", 10*Fix, f.X[0]-x0)
	}
}

// The heading octant, 0 east, counting clockwise on a
// screen with y down.
func TestOctantFollowsHeading(t *testing.T) {
	for _, c := range []struct {
		vx, vy int32
		want   uint8
	}{{100, 0, 0}, {70, 70, 1}, {0, 100, 2}, {-70, 70, 3}, {-100, 0, 4}, {-70, -70, 5}, {0, -100, 6}, {70, -70, 7}, {100, -30, 0}, {30, 100, 2},
		// either side of 22.5° (tan ≈ 106/256)
		{256, 105, 0}, {256, 106, 1}, {256, 107, 1}, {105, 256, 2}, {107, 256, 1}} {
		if got := Octant(c.vx, c.vy); got != c.want {
			t.Errorf("Octant(%d,%d) = %d, want %d", c.vx, c.vy, got, c.want)
		}
	}
}

// The rules are mirror-symmetric, and so must the integer arithmetic be: a
// flock and its mirror image (x → W−1−x, vx → −vx in a straight, centred
// box) stay exact mirrors step after step. A shift instead of a division
// on a negative offset or steering term rounds towards −∞ and breaks this
// (>>4 pushed the whole flock into the top-left corner).
func TestMirrorSymmetry(t *testing.T) {
	a, b := badge(t, 128), badge(t, 128)
	hx, hy := a.Box.W-1, a.Box.H-1
	for i := range a.N {
		b.X[i], b.Y[i], b.VX[i], b.VY[i] = hx-a.X[i], a.Y[i], -a.VX[i], a.VY[i]
	}
	c := badge(t, 128) // the y mirror
	for i := range a.N {
		c.X[i], c.Y[i], c.VX[i], c.VY[i] = a.X[i], hy-a.Y[i], a.VX[i], -a.VY[i]
	}
	for s := range 2000 {
		a.Step()
		b.Step()
		c.Step()
		for i := range a.N {
			if b.X[i] != hx-a.X[i] || b.Y[i] != a.Y[i] || b.VX[i] != -a.VX[i] || b.VY[i] != a.VY[i] {
				t.Fatalf("step %d boid %d: x mirror (%d,%d v %d,%d) of (%d,%d v %d,%d)", s, i, b.X[i], b.Y[i], b.VX[i], b.VY[i], a.X[i], a.Y[i], a.VX[i], a.VY[i])
			}
			if c.X[i] != a.X[i] || c.Y[i] != hy-a.Y[i] || c.VX[i] != a.VX[i] || c.VY[i] != -a.VY[i] {
				t.Fatalf("step %d boid %d: y mirror (%d,%d v %d,%d) of (%d,%d v %d,%d)", s, i, c.X[i], c.Y[i], c.VX[i], c.VY[i], a.X[i], a.Y[i], a.VX[i], a.VY[i])
			}
		}
	}
}

// Every boid steers from the flock as it was at the start of the step, so
// the order of the boids doesn't matter: the same flock stored backwards
// gives the same flock, backwards, step after step.
func TestOrderIndependent(t *testing.T) {
	a, b := badge(t, 128), badge(t, 128)
	n := a.N
	for i := range n {
		b.X[n-1-i], b.Y[n-1-i], b.VX[n-1-i], b.VY[n-1-i] = a.X[i], a.Y[i], a.VX[i], a.VY[i]
	}
	for s := range 500 {
		a.Step()
		b.Step()
		for i := range n {
			if b.X[n-1-i] != a.X[i] || b.Y[n-1-i] != a.Y[i] || b.VX[n-1-i] != a.VX[i] || b.VY[n-1-i] != a.VY[i] {
				t.Fatalf("step %d boid %d differs when the flock is stored backwards", s, i)
			}
		}
	}
}

// The three rules each act: a lone pair closer than Sep moves apart
// (separation), two boids flying apart at an angle turn towards each
// other's heading (alignment), and a boid with a neighbour ahead steers
// towards it (cohesion).
func TestRules(t *testing.T) {
	f := badge(t, 2)
	cx, cy := f.Box.CX, f.Box.CY
	// separation: one cell apart, both flying north
	f.X[0], f.Y[0], f.VX[0], f.VY[0] = cx, cy, 0, -f.VMin
	f.X[1], f.Y[1], f.VX[1], f.VY[1] = cx+Fix, cy, 0, -f.VMin
	gap := f.X[1] - f.X[0]
	f.Step()
	if f.X[1]-f.X[0] <= gap || f.VX[0] >= 0 || f.VX[1] <= 0 {
		t.Fatalf("separation: gap %d → %d, vx %d %d; want wider, moving apart", gap, f.X[1]-f.X[0], f.VX[0], f.VX[1])
	}
	// alignment: 4 cells apart along x (no separation; cohesion acts along x
	// only), headings east and north: each turns towards the other's
	// heading in y. Exactly: v0 = (128 + (0−128)/8 + 1024/64, (−128−0)/8) =
	// (128, −16), limited from octagonal 134 to 128: (122, −15); v1 = (0 +
	// 128/8 − 1024/64, −128 + 128/8) = (0, −112), within the limits.
	f.X[0], f.Y[0], f.VX[0], f.VY[0] = cx, cy, f.VMax, 0
	f.X[1], f.Y[1], f.VX[1], f.VY[1] = cx+4*Fix, cy, 0, -f.VMax
	f.Step()
	if f.VX[0] != 122 || f.VY[0] != -15 || f.VX[1] != 0 || f.VY[1] != -112 {
		t.Fatalf("alignment: v0 %d,%d v1 %d,%d; want 122,−15 and 0,−112", f.VX[0], f.VY[0], f.VX[1], f.VY[1])
	}
	// cohesion: same heading (north), a neighbour 5 cells to the east
	f.X[0], f.Y[0], f.VX[0], f.VY[0] = cx, cy, 0, -f.VMax
	f.X[1], f.Y[1], f.VX[1], f.VY[1] = cx+5*Fix, cy, 0, -f.VMax
	f.Step()
	if f.VX[0] <= 0 || f.VX[1] >= 0 {
		t.Fatalf("cohesion: vx %d %d; want each steering towards the other", f.VX[0], f.VX[1])
	}
	// cohesion in both axes: equal velocities (no alignment), the neighbour
	// 3 cells right and 3 down (no separation): exactly ±768/64 = ±12 each
	// way, within the speed limits (the y term was unpinned once)
	f.X[0], f.Y[0], f.VX[0], f.VY[0] = cx, cy, 70, 0
	f.X[1], f.Y[1], f.VX[1], f.VY[1] = cx+3*Fix, cy+3*Fix, 70, 0
	f.Step()
	if f.VX[0] != 82 || f.VY[0] != 12 || f.VX[1] != 58 || f.VY[1] != -12 {
		t.Fatalf("diagonal cohesion: v0 %d,%d v1 %d,%d; want 82,12 and 58,−12", f.VX[0], f.VY[0], f.VX[1], f.VY[1])
	}
}

// Spin raises the target angle by one every second Advance, and the angle
// follows one step per Advance.
func TestSpinRate(t *testing.T) {
	f := badge(t, 1)
	f.Box.Spin = true
	for range 10 {
		f.Box.Advance()
	}
	if f.Box.Target != 5 || f.Box.Angle != 5 {
		t.Fatalf("after 10 steps of spin: target %d angle %d, want 5 and 5", f.Box.Target, f.Box.Angle)
	}
}

// A boid thrown farther than the box is wide (a fast wall, a far jump) is
// put back inside, at the far wall, not reflected out the other side.
func TestFarThrowIsClampedIntoTheBox(t *testing.T) {
	f := badge(t, 1)
	f.X[0], f.Y[0], f.VX[0], f.VY[0] = f.Box.CX-2*f.Box.W, f.Box.CY, -f.VMax, 0
	f.walls(0)
	if bx, by := f.Box.ToBox(f.X[0], f.Y[0]); bx != f.Box.W-1 || by != f.Box.H/2 || f.VX[0] <= 0 {
		t.Fatalf("thrown two box widths left: box %d,%d v %d; want at the right wall %d, moving right", bx, by, f.VX[0], f.Box.W-1)
	}
}

// Seeding is deterministic, inside the box and between the speed limits'
// raw range; a different seed gives a different flock.
func TestSeed(t *testing.T) {
	a, b := badge(t, 64), badge(t, 64)
	b.Seed(7)
	c := badge(t, 64)
	c.Seed(7)
	same := true
	for i := range a.N {
		if bx, by := a.Box.ToBox(a.X[i], a.Y[i]); bx < 0 || by < 0 || bx >= a.Box.W || by >= a.Box.H {
			t.Fatalf("seeded boid %d outside the box: %d,%d", i, bx, by)
		}
		if b.X[i] != c.X[i] || b.VY[i] != c.VY[i] {
			t.Fatalf("seed 7 twice: boid %d differs", i)
		}
		same = same && a.X[i] == b.X[i]
	}
	if same {
		t.Fatal("seed 7 gives the default flock")
	}
}

// New refuses parameters it cannot run: no boids, an empty or too large
// box, a speed range upside down, a neighbour radius smaller than the
// separation distance.
func TestNewRefusesBadParams(t *testing.T) {
	p := DefaultParams()
	for _, c := range []struct {
		n    int
		w, h int32
		mod  func(*Params)
	}{
		{0, 64, 48, nil},
		{8, 0, 48, nil},
		{8, 64, -1, nil},
		{8, 300, 48, nil},
		{8, 64, 48, func(p *Params) { p.VMin, p.VMax = 130, 128 }},
		{8, 64, 48, func(p *Params) { p.VMin = 0 }},
		{8, 64, 48, func(p *Params) { p.R = p.Sep - 1 }},
		{8, 64, 48, func(p *Params) { p.Margin = -1 }},
	} {
		q := p
		if c.mod != nil {
			c.mod(&q)
		}
		if _, err := New(c.n, c.w, c.h, q); err == nil {
			t.Errorf("New(%d, %d, %d, %+v) accepted", c.n, c.w, c.h, q)
		}
	}
}

// Cell gives the boid's cell in the box (clamped to the box) and its
// heading octant on screen.
func TestCell(t *testing.T) {
	f := badge(t, 1)
	f.X[0], f.Y[0], f.VX[0], f.VY[0] = f.Box.CX+5*Fix+100, f.Box.CY-3*Fix, 0, f.VMax
	if x, y, o := f.Cell(0); x != 37 || y != 21 || o != 2 {
		t.Fatalf("Cell = %d,%d octant %d, want 37,21 octant 2 (south on screen)", x, y, o)
	}
	f.X[0] = f.Box.CX + 100*Fix // far outside: clamped to the last column
	if x, _, _ := f.Cell(0); x != 63 {
		t.Fatalf("Cell x %d outside the box, want clamped to 63", x)
	}
}

// Step allocates nothing (the embedded rule): a raw MemStats.Mallocs delta over
// 1,000 steps of the full flock while the box spins and shakes. Step is
// deterministic, so an allocation of its own shows in every measurement, while
// one by another goroutine of the test process does not repeat. The delta is
// therefore taken on one P, up to three times, and the test fails only when
// every attempt allocated.
func TestStepDoesNotAllocate(t *testing.T) {
	f := badge(t, 128)
	f.Box.Spin = true
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	var before, after runtime.MemStats
	var counts [3]uint64
	for a := range counts {
		runtime.ReadMemStats(&before)
		for s := range 1000 {
			if s%300 == 0 {
				f.Box.Shake(uint8(s), 10*Fix)
			}
			f.Box.Advance()
			f.Step()
		}
		runtime.ReadMemStats(&after)
		if counts[a] = after.Mallocs - before.Mallocs; counts[a] == 0 {
			return
		}
	}
	t.Fatalf("allocations in 1000 steps, each of %d attempts: %v", len(counts), counts)
}

// BenchmarkStep128: the largest badge flock, box spinning.
func BenchmarkStep128(b *testing.B) {
	f := badge(b, 128)
	f.Box.Spin = true
	for b.Loop() {
		f.Box.Advance()
		f.Step()
	}
}
