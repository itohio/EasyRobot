package rigid2d

import (
	"math"
	"runtime"
	"testing"
)

const dt = 1.0 / 240

// pingPong is a table-tennis-like ball: 20 mm radius, 2.7 g, a thin shell.
func pingPong(x, y, vx, vy, w float32) Ball {
	return Ball{X: x, Y: y, VX: vx, VY: vy, W: w, R: 0.02, Mass: 0.0027, InertiaK: 2.0 / 3}
}

func floor(m Material) Segment { return Segment{AX: -10, AY: 0, BX: 10, BY: 0, Material: m} }

// bounce steps w until the ball has touched something and moved away again,
// at most n steps.
func bounce(w *World, n int) {
	touched := false
	for i := 0; i < n; i++ {
		w.Step(dt)
		if w.Balls[0].Contacts > 0 {
			touched = true
		} else if touched {
			return
		}
	}
}

func energy(b *Ball) float64 {
	v2 := float64(b.VX)*float64(b.VX) + float64(b.VY)*float64(b.VY)
	i := float64(b.InertiaK) * float64(b.Mass) * float64(b.R) * float64(b.R)
	return 0.5*float64(b.Mass)*v2 + 0.5*i*float64(b.W)*float64(b.W)
}

// A ball hitting a still wall at 45°: a slippery wall leaves it with
// little spin; a grippy one makes it roll off (contact point at rest:
// ω·R equals the outgoing tangential speed).
func TestStillWallSpinDependsOnFriction(t *testing.T) {
	for _, c := range []struct {
		mu      float32
		rolling bool
	}{{0.02, false}, {0.8, true}} {
		w := World{Balls: []Ball{pingPong(0, 0.05, 1, -1, 0)}, Segments: []Segment{floor(Material{Restitution: 0.8, Friction: c.mu})}}
		bounce(&w, 1000)
		b := w.Balls[0]
		spin := math.Abs(float64(b.W * b.R)) // m/s at the rim
		if c.rolling {
			if math.Abs(spin-float64(b.VX)) > 0.01*float64(b.VX) {
				t.Errorf("grippy wall: rim speed %.4f, tangential speed %.4f; want rolling", spin, b.VX)
			}
		} else if spin > 0.1 {
			t.Errorf("slippery wall: rim speed %.4f m/s from a 1 m/s tangential hit, want < 0.1", spin)
		}
		if b.VY <= 0 {
			t.Errorf("mu %v: did not bounce: %+v", c.mu, b)
		}
	}
}

// Straight onto a frictionless wall: the normal speed comes back times the
// restitution, no spin.
func TestRestitution(t *testing.T) {
	for _, e := range []float32{0, 0.5, 1} {
		w := World{Balls: []Ball{pingPong(0, 0.05, 0, -2, 0)}, Segments: []Segment{floor(Material{Restitution: e})}}
		bounce(&w, 1000)
		if e == 0 {
			for i := 0; i < 50; i++ {
				w.Step(dt)
			}
		}
		b := w.Balls[0]
		if math.Abs(float64(b.VY-2*e)) > 1e-3 || b.W != 0 || b.VX != 0 {
			t.Errorf("e %v: %+v, want vy %v and no spin", e, b, 2*e)
		}
	}
}

// Coulomb: the tangential impulse never exceeds μ times the normal one, the
// contact point's sliding never reverses, and a still wall never adds
// energy; random angles, spins and materials.
func TestCoulombAndEnergyAtAStillWall(t *testing.T) {
	s := uint32(12345)
	rnd := func() float32 { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float32(s) / float32(^uint32(0)) }
	for i := 0; i < 2000; i++ {
		m := Material{Restitution: rnd(), Friction: 1.5 * rnd()}
		vx, vy, w0 := 4*rnd()-2, -0.2-3*rnd(), 400*rnd()-200
		b := pingPong(0, 0.0199, vx, vy, w0) // touching: resolved this step
		before := b
		wd := World{Balls: []Ball{b}, Segments: []Segment{floor(m)}}
		wd.Step(1e-6)
		a := wd.Balls[0]
		jn := float64(a.VY - before.VY)           // normal impulse / m
		jt := math.Abs(float64(a.VX - before.VX)) // tangential impulse / m
		if jt > float64(m.Friction)*jn+1e-5 {     // 1e-5: float32 resolution of velocities ≈ 2 m/s
			t.Fatalf("case %d %+v: tangential Δv %.5f > μ·normal Δv %.5f", i, m, jt, float64(m.Friction)*jn)
		}
		slip0 := float64(before.VX) + float64(before.W*before.R) // contact point (bottom) tangential speed
		slip1 := float64(a.VX) + float64(a.W*a.R)
		if slip0*slip1 < -1e-5 {
			t.Fatalf("case %d: sliding reversed %.5f → %.5f", i, slip0, slip1)
		}
		if energy(&a) > energy(&before)*(1+1e-5)+1e-12 {
			t.Fatalf("case %d %+v: energy %.3g → %.3g", i, m, energy(&before), energy(&a))
		}
	}
}

// A grippy paddle moving along its face imparts spin in proportion to its
// speed; a still one imparts none.
func TestMovingPaddleSpinProportionalToSpeed(t *testing.T) {
	spin := func(u float32) float32 {
		w := World{
			Balls: []Ball{pingPong(0.1, 0, -2, 0, 0)},
			Boxes: []Box{{MinX: -0.02, MinY: -0.1, MaxX: 0, MaxY: 0.1, VY: u, Material: Material{Restitution: 0.8, Friction: 1}}},
		}
		bounce(&w, 1000)
		return w.Balls[0].W
	}
	if s := spin(0); s != 0 {
		t.Fatalf("still paddle: spin %v", s)
	}
	s1, s2, s4 := spin(0.5), spin(1), spin(2)
	if s1 == 0 || math.Abs(float64(s2/s1-2)) > 0.01 || math.Abs(float64(s4/s1-4)) > 0.01 {
		t.Fatalf("spin at paddle speeds 0.5, 1, 2 m/s: %v %v %v, want 1:2:4", s1, s2, s4)
	}
	// A paddle at x ≤ 0 moving up (+y) drags the ball's left side up:
	// clockwise spin (ω < 0).
	if s1 >= 0 {
		t.Fatalf("spin %v, want clockwise (< 0)", s1)
	}
}

// Magnus: a ball sent away from the paddle (+x) curves towards +y when
// spinning counter-clockwise; with strong spin it curves round and comes
// back towards the paddle (x falls well below its farthest point), with
// weak spin it keeps going away.
func TestMagnusCurvesAndCanComeBack(t *testing.T) {
	// fly returns the farthest x, how far x then came back, and the end y.
	fly := func(w0 float32) (maxX, back, endY float32) {
		w := World{Params: Params{Magnus: 0.02, AngularDamping: 0.2}, Balls: []Ball{pingPong(0, 0, 2, 0, w0)}}
		for i := 0; i < int(3/dt); i++ {
			w.Step(dt)
			b := w.Balls[0]
			if b.X > maxX {
				maxX = b.X
			}
			back = max(back, maxX-b.X)
		}
		return maxX, back, w.Balls[0].Y
	}
	maxX, back, endY := fly(20)
	if back > 0 || endY <= 0 || maxX < 3 {
		t.Fatalf("weak spin: farthest x %v, came back %v, end y %v; want going away, curving to +y", maxX, back, endY)
	}
	if _, back, _ := fly(300); back < 0.3 {
		t.Fatalf("strong spin: came back %v m towards the paddle, want ≥ 0.3", back)
	}
	if _, _, endY := fly(-20); endY >= 0 { // clockwise curves the other way
		t.Fatalf("clockwise spin: end y %v, want < 0", endY)
	}
}

// Linear and angular damping: kinetic energy and spin fall every step, the
// spin as ω0·e^(−c·t).
func TestDampingDecay(t *testing.T) {
	w := World{Params: Params{Magnus: 0.02, LinearDamping: 0.5, AngularDamping: 1}, Balls: []Ball{pingPong(0, 0, 3, 1, 100)}}
	e := energy(&w.Balls[0])
	for i := 0; i < 480; i++ { // 2 s
		w.Step(dt)
		en := energy(&w.Balls[0])
		if en >= e {
			t.Fatalf("step %d: energy %v → %v", i, e, en)
		}
		e = en
	}
	if want := 100 * math.Exp(-2); math.Abs(float64(w.Balls[0].W)-want) > 1e-3*want {
		t.Fatalf("spin after 2 s %v, want %v", w.Balls[0].W, want)
	}
	// Magnus turns the velocity without changing its length, so the speed
	// falls as the linear damping alone: √10·e^(−0.5·2)
	b := w.Balls[0]
	if want := math.Sqrt(10) * math.Exp(-1); math.Abs(math.Hypot(float64(b.VX), float64(b.VY))-want) > 1e-3*want {
		t.Fatalf("speed after 2 s %v, want %v", math.Hypot(float64(b.VX), float64(b.VY)), want)
	}
}

// Changing the damping or the step between steps takes effect at once.
func TestDampingChangesTakeEffect(t *testing.T) {
	w := World{Balls: []Ball{pingPong(0, 0, 1, 0, 10)}}
	w.Step(dt)
	w.LinearDamping, w.AngularDamping = 2, 3
	w.Step(dt)
	if b := w.Balls[0]; math.Abs(float64(b.VX)-math.Exp(-2*dt)) > 1e-6 || math.Abs(float64(b.W)-10*math.Exp(-3*dt)) > 1e-5 {
		t.Fatalf("after switching damping on: %+v", b)
	}
	v, s := w.Balls[0].VX, w.Balls[0].W
	w.Step(2 * dt)
	if b := w.Balls[0]; math.Abs(float64(b.VX/v)-math.Exp(-4*dt)) > 1e-6 || math.Abs(float64(b.W/s)-math.Exp(-6*dt)) > 1e-6 {
		t.Fatalf("after doubling dt: %+v", b)
	}
	v, s = w.Balls[0].VX, w.Balls[0].W
	w.AngularDamping = 5 // only the angular coefficient changes
	w.Step(2 * dt)
	if b := w.Balls[0]; math.Abs(float64(b.VX/v)-math.Exp(-4*dt)) > 1e-6 || math.Abs(float64(b.W/s)-math.Exp(-10*dt)) > 1e-6 {
		t.Fatalf("after changing AngularDamping alone: %+v", b)
	}
}

// In the slip regime the tangential impulse is exactly Coulomb's limit:
// Δv_t = μ·(1+e)·|vn| (the normal impulse includes the restitution).
func TestSlidingFrictionIsCoulombsLimit(t *testing.T) {
	const mu, e = 0.1, 0.8
	w := World{Balls: []Ball{pingPong(0, 0.0199, 2, -1, 0)}, Segments: []Segment{floor(Material{Restitution: e, Friction: mu})}}
	w.Step(1e-6)
	b := w.Balls[0]
	if want := 2 - mu*(1+e)*1; math.Abs(float64(b.VX)-want) > 1e-5 || math.Abs(float64(b.VY)-e) > 1e-5 {
		t.Fatalf("after a sliding hit: vx %v vy %v, want vx %v (μ(1+e)|vn| taken off) and vy %v", b.VX, b.VY, want, e)
	}
}

// Segments are two-sided: Pong's top wall hit from below bounces the ball
// down.
func TestSegmentFromBelow(t *testing.T) {
	w := World{Balls: []Ball{pingPong(0.5, 0.5, 0.3, 1, 0)}, Segments: []Segment{{AX: 0, AY: 0.6, BX: 1, BY: 0.6, Material: Material{Restitution: 1}}}}
	bounce(&w, 1000)
	if b := w.Balls[0]; math.Abs(float64(b.VY+1)) > 1e-4 || b.Y > 0.6-b.R+1e-5 {
		t.Fatalf("%+v, want vy −1 below the wall", b)
	}
}

// A paddle rising into the top wall squeezes the ball against it: the ball
// must never be pushed through the wall (review E2: the paddle's push-out
// carried it past the wall's line, and the wall then pushed it out on the
// far side). The ball's centre stays at least R below the wall while the
// paddle keeps rising into and through the wall.
func TestSqueezedBallDoesNotTunnel(t *testing.T) {
	m := Material{Restitution: 0.9, Friction: 0.1}
	for _, vy := range []float32{0.3, 3} {
		w := World{
			Balls:    []Ball{pingPong(0.5, 0.55, 0, 0, 0)},
			Segments: []Segment{{AX: 0, AY: 0.6, BX: 1, BY: 0.6, Material: m}},
			Boxes:    []Box{{MinX: 0.45, MinY: 0.3, MaxX: 0.55, MaxY: 0.5, VY: vy, Material: Material{Restitution: 0.9, Friction: 0.9}}},
		}
		for i := 0; i < 400; i++ {
			w.Step(dt)
			if b := w.Balls[0]; b.Y > 0.6-b.R+1e-5 {
				t.Fatalf("paddle at %v m/s, step %d: ball at y %v, through the wall at 0.6 (box top %v)", vy, i, b.Y, w.Boxes[0].MaxY)
			}
		}
	}
}

// A ball whose centre lands exactly on a segment's line, or exactly on its
// end, is still pushed back to the side it came from (review E5: on the
// line it was always pushed to the left normal's side, i.e. through when
// coming from the right; on the end it got no contact).
func TestSegmentExactHits(t *testing.T) {
	const h = 1.0 / 256 // exact in float32: the step lands exactly on the target
	m := Material{Restitution: 1}
	// A→B along +x: the left normal points up; come from below onto the line
	w := World{Balls: []Ball{pingPong(0.5, -1.0/64, 0, 4, 0)}, Segments: []Segment{{AX: 0, AY: 0, BX: 1, BY: 0, Material: m}}}
	w.Step(h)
	if b := w.Balls[0]; b.Contacts != 1 || b.Y > -b.R+1e-6 || b.VY >= 0 {
		t.Fatalf("centre exactly on the line from below: %+v, want pushed back below with vy < 0", b)
	}
	// along the segment onto its ends A and B
	for _, c := range []struct{ x, vx float32 }{{-1.0 / 64, 4}, {1 + 1.0/64, -4}} {
		w = World{Balls: []Ball{pingPong(c.x, 0, c.vx, 0, 0)}, Segments: []Segment{{AX: 0, AY: 0, BX: 1, BY: 0, Material: m}}}
		w.Step(h)
		end := float32(0)
		if c.vx < 0 {
			end = 1
		}
		if b := w.Balls[0]; b.Contacts != 1 || math.Abs(float64(b.X-(end-b.R*c.vx/4))) > 1e-6 || b.VX != -c.vx {
			t.Fatalf("centre exactly on the end at x %v: %+v, want pushed back with vx %v", end, b, -c.vx)
		}
	}
	// onto B diagonally: pushed back along the way it came, 45° up and right
	w = World{Balls: []Ball{pingPong(1+1.0/64, 1.0/64, -4, -4, 0)}, Segments: []Segment{{AX: 0, AY: 0, BX: 1, BY: 0, Material: m}}}
	w.Step(h)
	if b, r := w.Balls[0], float64(w.Balls[0].R)/math.Sqrt2; b.Contacts != 1 || math.Abs(float64(b.X)-1-r) > 1e-6 || math.Abs(float64(b.Y)-r) > 1e-6 {
		t.Fatalf("centre exactly on B from up-right: %+v, want at (1+R/√2, R/√2)", b)
	}
	// starting on the line (no side to come from), it is pushed out on the
	// side it moves to
	w = World{Balls: []Ball{pingPong(0.5, 0, 0, -1, 0)}, Segments: []Segment{{AX: 0, AY: 0, BX: 1, BY: 0, Material: m}}}
	w.Step(h)
	if b := w.Balls[0]; math.Abs(float64(b.Y+b.R)) > 1e-6 {
		t.Fatalf("starting on the line, moving down: %+v, want pushed out below (y = −R)", b)
	}
}

// A wall moving fast enough to pass the ball's centre in one step sweeps
// the ball along (the side is taken against where the wall was at the
// start of the step), it does not pass over it.
func TestFastWallSweepsTheBall(t *testing.T) {
	w := World{
		Balls:    []Ball{pingPong(0.5, 0, 0, 0, 0)},
		Segments: []Segment{{AX: 0, AY: 0.03, BX: 1, BY: 0.03, VY: -12, Material: Material{Restitution: 0.5}}}, // 0.05 m per step
	}
	w.Step(dt)
	if b, s := w.Balls[0], w.Segments[0]; b.Y > s.AY-b.R+1e-5 || b.VY > -12 {
		t.Fatalf("wall at y %v after the step: ball %+v, want below it moving down faster than the wall", s.AY, b)
	}
}

// A paddle moving along x (its top face, along its length) drags the ball's
// bottom forward: the ball gains vx and counter-clockwise spin until its
// contact point moves with the box (vx + ω·R = 1 m/s).
func TestBoxMovingAlongX(t *testing.T) {
	w := World{Balls: []Ball{pingPong(0, 0.05, 0, -1, 0)}, Boxes: []Box{{MinX: -1, MinY: -0.02, MaxX: 1, MaxY: 0, VX: 1, Material: Material{Restitution: 0.5, Friction: 1}}}}
	bounce(&w, 1000)
	if b := w.Balls[0]; b.VX <= 0 || b.W <= 0 || math.Abs(float64(b.VX+b.W*b.R-1)) > 0.01 {
		t.Fatalf("%+v: want vx > 0, counter-clockwise spin, contact point at the box's 1 m/s", b)
	}
}

// Magnus turns the velocity by 2·atan(θ/2) per step, θ = Magnus·ω·dt (the
// Cayley rotation), and gravity accelerates along both axes.
func TestMagnusTurnAndGravityAxes(t *testing.T) {
	w := World{Params: Params{Magnus: 0.5}, Balls: []Ball{pingPong(0, 0, 1, 0, 100)}}
	w.Step(0.01)
	th := 0.5 * 100 * 0.01
	if got, want := math.Atan2(float64(w.Balls[0].VY), float64(w.Balls[0].VX)), 2*math.Atan(th/2); math.Abs(got-want) > 1e-6 {
		t.Fatalf("turn %v rad in one step, want %v", got, want)
	}
	w = World{Params: Params{GX: 3, GY: -4}, Balls: []Ball{pingPong(0, 0, 0, 0, 0)}}
	for i := 0; i < 100; i++ {
		w.Step(0.01)
	}
	if b := w.Balls[0]; math.Abs(float64(b.VX)-3) > 1e-4 || math.Abs(float64(b.VY)+4) > 1e-4 {
		t.Fatalf("after 1 s of gravity (3, −4): v (%v, %v)", b.VX, b.VY)
	}
}

// Gravity onto a floor: the ball comes to rest on it, not in it.
func TestRestsOnTheFloor(t *testing.T) {
	w := World{Params: Params{GY: -9.81}, Balls: []Ball{pingPong(0, 0.3, 0, 0, 0)}, Segments: []Segment{floor(Material{Restitution: 0.5, Friction: 0.3})}}
	for i := 0; i < int(5/dt); i++ {
		w.Step(dt)
	}
	b := w.Balls[0]
	if math.Abs(float64(b.Y-b.R)) > 5e-4 || math.Abs(float64(b.VY)) > 0.05 {
		t.Fatalf("after 5 s: y %v (radius %v), vy %v", b.Y, b.R, b.VY)
	}
}

// Overlaps are pushed out along the shortest way: off a box corner
// diagonally, and out of a box the centre has entered.
func TestBoxContactPushesOut(t *testing.T) {
	box := Box{MinX: 0, MinY: 0, MaxX: 1, MaxY: 0.2, Material: Material{Restitution: 0.5}}
	w := World{Balls: []Ball{pingPong(1.01, 0.21, 0, 0, 0)}, Boxes: []Box{box}}
	w.Step(dt)
	b := w.Balls[0]
	if d := math.Hypot(float64(b.X-1), float64(b.Y-0.2)); math.Abs(d-float64(b.R)) > 1e-5 || math.Abs(float64(b.X-1-(b.Y-0.2))) > 1e-5 {
		t.Fatalf("corner: centre at (%v, %v), distance %v, want %v along the diagonal", b.X, b.Y, d, b.R)
	}
	w = World{Balls: []Ball{pingPong(0.5, 0.19, 0, 0, 0)}, Boxes: []Box{box}}
	w.Step(dt)
	if b := w.Balls[0]; math.Abs(float64(b.Y-0.2-b.R)) > 1e-5 || b.X != 0.5 {
		t.Fatalf("inside: centre at (%v, %v), want (0.5, %v)", b.X, b.Y, 0.2+b.R)
	}
}

// Head-on into a segment's end (along its direction): the end acts as a
// point, the ball comes straight back.
func TestSegmentEnd(t *testing.T) {
	for _, c := range []struct{ x, vx float32 }{{-0.1, 1}, {1.1, -1}} { // towards A, towards B
		w := World{Balls: []Ball{pingPong(c.x, 0, c.vx, 0, 0)}, Segments: []Segment{{AX: 0, BX: 1, Material: Material{Restitution: 1, Friction: 0.5}}}}
		bounce(&w, 1000)
		if b := w.Balls[0]; math.Abs(float64(b.VX+c.vx)) > 1e-4 || math.Abs(float64(b.VY)) > 1e-6 || b.W != 0 {
			t.Fatalf("from x %v: %+v, want vx %v", c.x, b, -c.vx)
		}
	}
}

// A ball dropped onto a grippy belt moving +x leaves rolling on the belt:
// its contact point moves with the belt (vx + ω·R = belt speed).
func TestConveyorFloor(t *testing.T) {
	belt := floor(Material{Restitution: 0.5, Friction: 1})
	belt.VX = 0.8
	w := World{Balls: []Ball{pingPong(0, 0.05, 0, -1, 0)}, Segments: []Segment{belt}}
	bounce(&w, 1000)
	b := w.Balls[0]
	if b.VX <= 0 || math.Abs(float64(b.VX+b.W*b.R-0.8)) > 0.008 {
		t.Fatalf("%+v: contact point speed %v, want the belt's 0.8", b, b.VX+b.W*b.R)
	}
}

// An overlap with a ball already moving away is pushed out without an
// impulse.
func TestSeparatingContactKeepsVelocity(t *testing.T) {
	w := World{Balls: []Ball{pingPong(0, 0.01, 0.3, 1, 5)}, Segments: []Segment{floor(Material{Restitution: 1, Friction: 1})}}
	w.Step(1e-6)
	if b := w.Balls[0]; b.VX != 0.3 || b.VY != 1 || b.W != 5 || math.Abs(float64(b.Y-b.R)) > 1e-6 {
		t.Fatalf("%+v, want velocity and spin unchanged, y = R", b)
	}
}

// The kinematic shapes move with their velocity each step.
func TestKinematicShapesMove(t *testing.T) {
	w := World{Boxes: []Box{{MaxX: 1, MaxY: 1, VY: 2}}, Segments: []Segment{{BX: 1, VX: -1}}}
	for i := 0; i < 240; i++ {
		w.Step(dt)
	}
	if bx := w.Boxes[0]; math.Abs(float64(bx.MinY-2)) > 1e-4 || math.Abs(float64(bx.MaxY-3)) > 1e-4 {
		t.Fatalf("box %+v after 1 s at 2 m/s", bx)
	}
	if s := w.Segments[0]; math.Abs(float64(s.AX+1)) > 1e-4 || math.Abs(float64(s.BX)) > 1e-4 {
		t.Fatalf("segment %+v after 1 s at −1 m/s", s)
	}
}

// pong is a table with walls on all four sides (the side walls sit behind
// the paddles, so the ball stays in play) and two moving paddles.
func pong() World {
	m := Material{Restitution: 0.9, Friction: 0.1}
	back := Material{Restitution: 1}
	return World{
		Params: Params{Magnus: 0.02, LinearDamping: 0.05, AngularDamping: 0.3},
		Balls:  []Ball{pingPong(0.5, 0.3, 1.7, 0.9, 40)},
		Segments: []Segment{
			{AX: 0, AY: 0, BX: 1, BY: 0, Material: m}, {AX: 0, AY: 0.6, BX: 1, BY: 0.6, Material: m},
			{AX: -0.06, AY: 0, BX: -0.06, BY: 0.6, Material: back}, {AX: 1.08, AY: 0, BX: 1.08, BY: 0.6, Material: back},
		},
		Boxes: []Box{
			{MinX: -0.02, MinY: 0.2, MaxX: 0, MaxY: 0.4, VY: 0.3, Material: Material{Restitution: 0.9, Friction: 0.9}},
			{MinX: 1, MinY: 0.2, MaxX: 1.02, MaxY: 0.4, VY: -0.3, Material: Material{Restitution: 0.9, Friction: 0.9}},
		},
	}
}

// Same start, same steps: bit-identical state.
func TestDeterministic(t *testing.T) {
	a, b := pong(), pong()
	for i := 0; i < 5000; i++ {
		a.Step(dt)
		b.Step(dt)
	}
	if a.Balls[0] != b.Balls[0] {
		t.Fatalf("%+v != %+v", a.Balls[0], b.Balls[0])
	}
}

// Step allocates nothing (the owner's rule), measured where allocations
// could hide: two balls resting under gravity, one on a segment and one on
// a box, so both contact paths apply an impulse every step, and a dt that
// changes every step, so the damping cache refreshes every step. A raw
// MemStats.Mallocs delta over 1,000 steps, not AllocsPerRun (its integer
// division hides fewer than one allocation per run). Review E1.
func TestStepDoesNotAllocate(t *testing.T) {
	m := Material{Restitution: 0.5, Friction: 0.3}
	w := World{
		// no linear damping, so a vy above v + g·h can only come from an
		// impulse; the angular damping still refreshes the cache
		Params:   Params{GY: -9.81, Magnus: 0.02, AngularDamping: 0.3},
		Balls:    []Ball{pingPong(0, 0.02, 0, 0, 5), pingPong(5, 1.02, 0.1, 0, -5)},
		Segments: []Segment{floor(m)},
		Boxes:    []Box{{MinX: 4, MinY: 0.5, MaxX: 6, MaxY: 1, Material: m}},
	}
	h := func(i int) float32 { return dt * (1 + float32(i%2)/100) }
	for i := 0; i < 10; i++ { // settle onto the surfaces
		w.Step(h(i))
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	impulses := 0
	for i := 0; i < 1000; i++ {
		v0, v1 := w.Balls[0].VY, w.Balls[1].VY
		w.Step(h(i))
		if w.Balls[0].VY > v0+w.GY*h(i)+1e-6 && w.Balls[1].VY > v1+w.GY*h(i)+1e-6 {
			impulses++
		}
	}
	runtime.ReadMemStats(&after)
	if impulses != 1000 {
		t.Fatalf("%d of 1000 steps had an impulse on both balls; the scene must exercise the impulse path every step", impulses)
	}
	if n := after.Mallocs - before.Mallocs; n != 0 {
		t.Fatalf("%d allocations in 1000 steps", n)
	}
}

// stepBench steps a scene b.N times, restarting it every 10 s of simulated
// time (else damping stops the ball and the paddles drift away, and a long
// run would time an idle world). It reports the share of steps with a
// contact, so the time is read against the contact work it includes.
func stepBench(b *testing.B, scene func() World) {
	w, start := scene(), scene()
	contacts, steps := 0, 0
	for b.Loop() {
		if steps%2400 == 0 {
			copy(w.Balls, start.Balls)
			copy(w.Segments, start.Segments)
			copy(w.Boxes, start.Boxes)
		}
		w.Step(dt)
		steps++
		if w.Balls[0].Contacts > 0 {
			contacts++
		}
	}
	b.ReportMetric(float64(contacts)/float64(steps), "contact-steps/step")
}

// BenchmarkStepPong: one ball in play on a walled table with two moving
// paddles (4 segments, 2 boxes), replaying its first 10 s.
func BenchmarkStepPong(b *testing.B) { stepBench(b, pong) }

// BenchmarkStepInContact: a ball resting on a floor under gravity, a
// contact resolved every step.
func BenchmarkStepInContact(b *testing.B) {
	stepBench(b, func() World {
		return World{Params: Params{GY: -9.81}, Balls: []Ball{pingPong(0, 0.02, 0, 0, 0)}, Segments: []Segment{floor(Material{Restitution: 0.5, Friction: 0.3})}}
	})
}
