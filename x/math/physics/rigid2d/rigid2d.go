// Package rigid2d is a small 2D rigid-body brick for games and demos: balls
// (circles with spin) moving among kinematic walls (line segments) and boxes
// (axis-aligned rectangles, e.g. paddles) that move at a set velocity and are
// not pushed back.
//
// Each Step (fixed dt, semi-implicit Euler) rotates every ball's velocity by
// the Magnus effect (acceleration Magnus·ω·perp(v), i.e. ∝ ω × v; the turn
// per step is 2·atan(θ/2), θ = Magnus·ω·dt, a rotation that keeps |v| in
// exact arithmetic — float32 drifts about 0.5 % per 100k steps — so lift
// does no work), adds gravity, applies
// linear and angular damping (e^(−c·dt)), moves the ball, and resolves its
// contacts: the overlap is pushed out along the contact normal, and an
// approaching contact gets a normal impulse with the surface's restitution
// and a tangential impulse that would stop the contact point sliding,
// limited by Coulomb friction (μ·normal impulse) of the surface. The
// tangential impulse also changes the spin (I = InertiaK·m·R²), so a
// slippery wall leaves the spin nearly unchanged and a grippy moving paddle
// imparts spin in proportion to its speed along its face.
//
// Limits: no ball–ball contacts; contacts are discrete, so a ball should not
// travel more than about its radius per step (use a smaller dt or substeps
// for faster balls). Walls win over paddles: a ball squeezed between a
// paddle and a wall stays on its side of the wall and may overlap the
// paddle.
//
// float32, deterministic for a given sequence of steps, no allocation.
package rigid2d

import "github.com/chewxy/math32"

// Material is a surface's contact behaviour.
type Material struct {
	Restitution float32 // 0 (dead) … 1 (elastic), normal speed kept after a hit
	Friction    float32 // Coulomb μ ≥ 0
}

// Ball is a circle with spin. Units are the caller's (SI suggested: m, m/s,
// rad/s, kg).
type Ball struct {
	X, Y, VX, VY float32
	W            float32 // angular velocity, counter-clockwise positive
	R, Mass      float32
	// InertiaK sets the moment of inertia I = InertiaK·Mass·R²: ½ disc,
	// ⅖ solid sphere, ⅔ thin spherical shell (a table-tennis ball). ≤ 0
	// means contacts never change the spin; a W set by the caller is kept
	// and still drives Magnus and the slip.
	InertiaK float32
	// Contacts counts the contacts resolved in the last Step (e.g. for a
	// sound on a hit). A ball resting on a surface touches it every step
	// (Contacts 1, with a small micro-bounce, |VY| ≈ 0.02 m/s at 240 Hz).
	Contacts uint8
}

// Segment is a kinematic two-sided wall from A to B moving at (VX, VY).
type Segment struct {
	AX, AY, BX, BY float32
	VX, VY         float32
	Material
}

// Box is a kinematic axis-aligned rectangle moving at (VX, VY).
type Box struct {
	MinX, MinY, MaxX, MaxY float32
	VX, VY                 float32
	Material
}

// Params are the world's forces.
type Params struct {
	GX, GY         float32 // gravity (acceleration)
	Magnus         float32 // lift per spin: a = Magnus·ω·(−vy, vx), 1/rad
	LinearDamping  float32 // 1/s: v decays as e^(−c·t)
	AngularDamping float32 // 1/s: ω decays as e^(−c·t)
}

// World holds the bodies; the slices are the caller's and Step does not
// grow them.
type World struct {
	Params
	Balls    []Ball
	Segments []Segment
	Boxes    []Box

	damp damping
}

// damping caches e^(−c·dt) for the last dt and coefficients: two exp calls
// per step are most of a step's cost on a core without an FPU (RP2040).
type damping struct {
	dt, linC, angC, lin, ang float32
	ok                       bool
}

func (d *damping) factors(dt, linC, angC float32) (lin, ang float32) {
	if !d.ok || d.dt != dt || d.linC != linC || d.angC != angC {
		*d = damping{dt: dt, linC: linC, angC: angC, lin: math32.Exp(-linC * dt), ang: math32.Exp(-angC * dt), ok: true}
	}
	return d.lin, d.ang
}

// Step advances the world by dt.
func (w *World) Step(dt float32) {
	for i := range w.Segments {
		s := &w.Segments[i]
		s.AX += s.VX * dt
		s.AY += s.VY * dt
		s.BX += s.VX * dt
		s.BY += s.VY * dt
	}
	for i := range w.Boxes {
		b := &w.Boxes[i]
		b.MinX += b.VX * dt
		b.MaxX += b.VX * dt
		b.MinY += b.VY * dt
		b.MaxY += b.VY * dt
	}
	lin, ang := w.damp.factors(dt, w.LinearDamping, w.AngularDamping)
	for i := range w.Balls {
		b := &w.Balls[i]
		px, py := b.X, b.Y // where the ball starts this step
		if w.Magnus != 0 && b.W != 0 {
			// rotate v by θ = Magnus·ω·dt (Cayley form: exact length, no trig)
			th := w.Magnus * b.W * dt
			q := 1 + th*th/4
			c, s := (1-th*th/4)/q, th/q
			b.VX, b.VY = c*b.VX-s*b.VY, s*b.VX+c*b.VY
		}
		b.VX = (b.VX + w.GX*dt) * lin
		b.VY = (b.VY + w.GY*dt) * lin
		b.W *= ang
		b.X += b.VX * dt
		b.Y += b.VY * dt
		b.Contacts = 0
		// Boxes first, segments last: a paddle that squeezes the ball against
		// a wall may push it into the wall, and the wall, resolved after it,
		// pushes it back to the side it started on, so walls are never
		// crossed (the ball may be left overlapping the paddle instead).
		for j := range w.Boxes {
			b.box(&w.Boxes[j])
		}
		for j := range w.Segments {
			b.segment(&w.Segments[j], px, py, dt)
		}
	}
}

// segment resolves a contact with s. Against the segment's interior the
// normal is the line's own (a closest point computed on a long segment
// would tilt it by rounding), pointing to the side the ball was on at the
// start of the step (px, py, against the segment where it was then), so a
// ball pushed onto or across the line this step is pushed back, not
// through. Against an end the normal points from the end; a ball centred
// exactly on the end is pushed back the way it came.
func (b *Ball) segment(s *Segment, px, py, dt float32) {
	ex, ey := s.BX-s.AX, s.BY-s.AY
	l2 := ex*ex + ey*ey
	rx, ry := b.X-s.AX, b.Y-s.AY
	// the ball's start relative to A at the start of the step
	sx, sy := px-(s.AX-s.VX*dt), py-(s.AY-s.VY*dt)
	if l2 > 0 {
		if t := rx*ex + ry*ey; t > 0 && t < l2 {
			c := ry*ex - rx*ey // l × the distance along the left normal
			side := sy*ex - sx*ey
			if side == 0 {
				side = c // started on the line: the side it is on now
			}
			l := math32.Sqrt(l2)
			nx, ny, d := -ey/l, ex/l, c/l // left normal
			if side < 0 {
				nx, ny, d = -nx, -ny, -d
			}
			if d < b.R {
				b.contact(nx, ny, b.R-d, s.VX, s.VY, s.Material)
			}
			return
		}
		if rx*ex+ry*ey >= l2 {
			rx, ry = b.X-s.BX, b.Y-s.BY
			sx, sy = sx-ex, sy-ey // start relative to B
		}
	}
	d2 := rx*rx + ry*ry
	if d2 >= b.R*b.R {
		return
	}
	if d2 == 0 { // centred on the end: back the way it came
		rx, ry = sx, sy
		if d2 = rx*rx + ry*ry; d2 == 0 {
			return // it has not moved relative to the end: no direction
		}
		n := math32.Sqrt(d2)
		b.contact(rx/n, ry/n, b.R, s.VX, s.VY, s.Material)
		return
	}
	d := math32.Sqrt(d2)
	b.contact(rx/d, ry/d, b.R-d, s.VX, s.VY, s.Material)
}

// box resolves a contact with x.
func (b *Ball) box(x *Box) {
	cx, cy := min(max(b.X, x.MinX), x.MaxX), min(max(b.Y, x.MinY), x.MaxY)
	nx, ny := b.X-cx, b.Y-cy
	d2 := nx*nx + ny*ny
	if d2 >= b.R*b.R {
		return
	}
	if d2 > 0 {
		d := math32.Sqrt(d2)
		b.contact(nx/d, ny/d, b.R-d, x.VX, x.VY, x.Material)
		return
	}
	// centre inside: out through the nearest face
	l, r, dn, up := b.X-x.MinX, x.MaxX-b.X, b.Y-x.MinY, x.MaxY-b.Y
	switch m := min(l, r, dn, up); m {
	case up:
		b.contact(0, 1, up+b.R, x.VX, x.VY, x.Material)
	case dn:
		b.contact(0, -1, dn+b.R, x.VX, x.VY, x.Material)
	case l:
		b.contact(-1, 0, l+b.R, x.VX, x.VY, x.Material)
	default:
		b.contact(1, 0, r+b.R, x.VX, x.VY, x.Material)
	}
}

// contact pushes b out by depth along the unit normal (nx, ny) (from the
// surface to the ball) and applies the impulse of a surface moving at
// (ux, uy) with material m.
func (b *Ball) contact(nx, ny, depth, ux, uy float32, m Material) {
	b.X += nx * depth
	b.Y += ny * depth
	if b.Contacts < 255 {
		b.Contacts++
	}
	// velocity of the ball's contact point (r = −R·n; ω × r = ω·R·(ny, −nx))
	// relative to the surface
	rvx := b.VX + b.W*b.R*ny - ux
	rvy := b.VY - b.W*b.R*nx - uy
	vn := rvx*nx + rvy*ny
	if vn >= 0 {
		return // separating or resting
	}
	jn := -(1 + m.Restitution) * vn // impulse / mass
	tx, ty := -ny, nx
	vt := rvx*tx + rvy*ty
	// a tangential impulse jt (per mass) changes the contact point's
	// tangential speed by jt·(1 + 1/InertiaK): jt = −vt/kt stops the slip
	kt := float32(1)
	if b.InertiaK > 0 {
		kt += 1 / b.InertiaK
	}
	lim := m.Friction * jn
	jt := min(max(-vt/kt, -lim), lim)
	b.VX += jn*nx + jt*tx
	b.VY += jn*ny + jt*ty
	if b.InertiaK > 0 && b.R > 0 {
		b.W -= jt / (b.InertiaK * b.R) // Δω = (r × J)/I = −R·m·jt / (k·m·R²)
	}
}
