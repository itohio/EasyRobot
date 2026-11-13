// Package flip is a 2D liquid: particles that carry the liquid's
// velocity, moved over a coarse staggered (MAC) grid where the velocity
// field is made divergence-free each step (PIC/FLIP, after Zhu & Bridson
// 2005 and Müller's "FLIP water" notes). The liquid keeps momentum
// (sloshes, waves travel and reflect, splashes), its particle count never
// changes, and at rest its surface lies perpendicular to gravity.
//
// Integer only, for cores without an FPU (RP2040). The arithmetic is int32:
// velocities, gravity and parameters in 1/65536 of a cell (One), transfer
// weights in 1/256. Storage is small, for a badge with tens of kilobytes
// free: particle positions in 1/256 of a cell (Cell) and velocities in
// 1/16384 of a cell per step (VOne), 16 bits each; per grid face a 32-bit
// velocity and a 16-bit copy; per cell a byte and two 16-bit words. New
// allocates Size(w, h, n) bytes: 21,094 for 32×24 cells and 700
// particles, 16,056 for 28×21 and 524. One Step is one time step;
// velocities are in cells per step and gravity in cells per step². No
// allocation after New.
package flip

import "errors"

// One is one cell (or one cell per step) in the arithmetic, gravity and
// Params: Q16.
const One = 1 << 16

// Cell is one cell in particle positions (X, Y): Q8.
const Cell = 1 << 8

// VOne is one cell per step in particle velocities (VX, VY): Q14.
const VOne = 1 << 14

// ErrParams reports a size New cannot run.
var ErrParams = errors.New("flip: invalid parameters")

// Params tune the solver. DefaultParams suits the badge.
type Params struct {
	Iterations int   // Gauss–Seidel sweeps of the incompressibility solve
	Separate   int   // passes pushing overlapping particles apart
	FlipQ8     int32 // FLIP share of the velocity update, /256 (the rest is PIC)
	OverQ8     int32 // over-relaxation of the solve, /256
	MaxV       int32 // speed limit, Q16 cells per step (keeps a particle within a cell per step)
	Calm       int32 // below this speed (Q16 cells per step) FLIP fades into PIC; 0 = never
	RestQ8     int32 // a particle slower than RestQ8/256 of gravity's pull per step does not move; 0 = always moves
	HoldQ8     int32 // a still particle keeps HoldQ8/256 per step of a velocity against its last move (see advect); 256 = all
	Stick      int32 // a liquid cell stays liquid while a particle is this close to it (Q16 cells; see toGrid)
	DriftQ8    int32 // outflow from cells packed denser than Reset packs them, /256 of a cell per step per rest density; 0 = off
	Radius     int32 // particle radius, Q16 cells
}

// DefaultParams: 20 sweeps, 2 separation passes, 90 % FLIP fading into
// PIC below 1/16 cell per step, particles at rest below 0.8 of a step's
// gravity, a still particle keeping 7/8 per step of a push back against
// its last move, liquid cells sticky within 3/16 cell, drift 1/4, over-
// relaxation 1.9, particle radius 0.3 cell, speed limit 0.8 cell per step.
// Tuned for gravity of 0.01–0.02 cell per step² (GX, GY up to 1311): for
// faster motion take more steps per frame rather than stronger gravity
// (at 0.035 a still surface flickers again).
func DefaultParams() Params {
	return Params{Iterations: 20, Separate: 2, FlipQ8: 230, OverQ8: 486, MaxV: One * 8 / 10, Calm: One / 16, RestQ8: 200, HoldQ8: 224, Stick: One * 3 / 16, DriftQ8: 64, Radius: One * 3 / 10}
}

// Cell bytes: the type in the low bits, and whether the cell was liquid
// last step (see toGrid).
const (
	air uint8 = iota
	fluid
	solid

	kind     = 3      // mask of the type
	wasFluid = 1 << 2 // the cell was liquid last step
)

// Liquid is a box of W×H cells (the outer ring is wall) holding len(X)
// particles. X, Y are the particles' positions (1/Cell of a cell), VX, VY
// their velocities (1/VOne of a cell per step); GX, GY is gravity (Q16
// cells per step², see DefaultParams for the range), which may point
// anywhere — an accelerometer's in-plane reading scaled to it. The caller
// may move particles (they are kept inside the walls) and set gravity
// between steps.
type Liquid struct {
	prm    Params // fixed at New
	W, H   int
	X, Y   []uint16
	VX, VY []int16
	GX, GY int32

	u, v   []int32  // face velocities (Q16): u at (i, j+½) on (W+1)×H, v at (i+½, j) on W×(H+1)
	pu, pv []int16  // during the transfer its weight sums (Q8), then the face velocities before the solve (Q14)
	cell   []uint8  // air, fluid or solid, and wasFluid
	aux    []uint16 // particles around each cell centre (Q8 weights) for the solve; then the buckets' start for separate
	pres   []int16  // pressure impulse per cell (Q16), kept between steps (warm start), saturating
	order  []uint16 // particles bucketed by cell for separate
	last   []uint8  // direction bits (movedR…) of each particle's last move (see advect)
	lastGX int32    // gravity when last was last cleared (see advect)
	lastGY int32
	rest   int32 // aux of a cell packed as Reset packs it
	drift  int32 // outflow (Q16) per rest density of excess, ×256
	r      int32 // particle radius, Q8
	stick  int32 // Params.Stick, Q8
	calm   int32 // 2²⁴/Calm, 0 when Calm is 0
}

// Size is the number of bytes of the arrays New allocates for a w×h box
// of n particles: 11 per particle, 6 per grid face and 5 per cell. The
// Liquid itself adds about 310 more (32-bit TinyGo: New took 21,408 B at
// 32×24 with 700 particles, where Size is 21,094).
func Size(w, h, n int) int {
	return n*(2+2+2+2+2+1) + ((w+1)*h+w*(h+1))*(4+2) + w*h*(1+2+2) + 2
}

// New makes a w×h-cell box (3 ≤ w, h ≤ 255; the border is wall) holding n
// particles (1 ≤ n ≤ 65535) at rest on the floor (see Reset). It is the
// only call that allocates: the Liquid and Size(w, h, n) bytes of arrays.
func New(w, h, n int, p Params) (*Liquid, error) {
	if w < 3 || h < 3 || w > 255 || h > 255 || n < 1 || n > 65535 || p.Iterations < 0 || p.Separate < 0 ||
		p.Radius < One/8 || p.Radius >= One/2 || p.MaxV <= 0 || p.MaxV > One ||
		p.FlipQ8 < 0 || p.FlipQ8 > 256 || p.OverQ8 < 0 || p.OverQ8 > 512 ||
		(p.Calm != 0 && (p.Calm < One/64 || p.Calm > One)) || p.RestQ8 < 0 || p.RestQ8 > 1024 ||
		p.HoldQ8 < 0 || p.HoldQ8 > 256 ||
		p.Stick < 0 || p.Stick > One/2 || p.DriftQ8 < 0 || p.DriftQ8 > 1024 {
		return nil, ErrParams
	}
	l := &Liquid{prm: p, W: w, H: h, r: p.Radius >> 8, stick: p.Stick >> 8}
	l.X, l.Y = make([]uint16, n), make([]uint16, n)
	l.VX, l.VY = make([]int16, n), make([]int16, n)
	l.order, l.last = make([]uint16, n), make([]uint8, n)
	l.u, l.pu = make([]int32, (w+1)*h), make([]int16, (w+1)*h)
	l.v, l.pv = make([]int32, w*(h+1)), make([]int16, w*(h+1))
	l.cell, l.aux, l.pres = make([]uint8, w*h), make([]uint16, w*h+1), make([]int16, w*h)
	if p.Calm > 0 {
		l.calm = (1 << 24) / p.Calm
	}
	if p.DriftQ8 > 0 {
		// hexagonal packing 2.2 radii apart (the radius as positions hold
		// it, as Reset packs): 1/(d²·√3/2) particles per cell
		d := int64(l.r) << 8 * 22 / 10 // Q16
		l.rest = int32(int64(256) << 32 / (d * d * 866 / 1000))
		l.drift = One * p.DriftQ8 / l.rest
	}
	l.Reset()
	return l, nil
}

// Reset puts every particle back at rest in rows from the floor (+y) up,
// hexagonally packed 2.2 radii apart (about how a shaken liquid settles,
// so the level does not change once it has been thrown around).
func (l *Liquid) Reset() {
	r := l.r
	d := 2 * r * 11 / 10
	lo, hiX, hiY := Cell+r, int32(l.W-1)*Cell-r, int32(l.H-1)*Cell-r
	x, y, row := lo, hiY, 0
	for i := range l.X {
		l.X[i], l.Y[i], l.VX[i], l.VY[i] = uint16(x), uint16(y), 0, 0
		x += d
		if x > hiX {
			row++
			x = lo + int32(row&1)*d/2
			y -= d * 87 / 100 // hexagonal rows
			if y < lo {
				y = lo
			}
		}
	}
	clear(l.last)
	clear(l.cell)
	clear(l.pres)
}

// Step advances the liquid one time step: particle velocities go to the
// grid, gravity is added there, the grid is made divergence-free, the
// particles take the result back (PIC/FLIP) and move with it, then
// overlapping particles are pushed apart and kept inside the walls.
// Gravity is balanced by the solve before anything moves, so a liquid at
// rest stays at rest.
func (l *Liquid) Step() {
	l.walls() // particles the caller moved
	l.toGrid()
	l.gravity()
	l.solve()
	l.fromGrid()
	l.advect()
	if l.prm.Separate > 0 {
		l.bucket()
		for range l.prm.Separate {
			l.separate()
		}
	}
	l.walls()
}

// wet reports whether cell k is liquid.
func (l *Liquid) wet(k int) bool { return l.cell[k]&kind == fluid }

// gravity adds GX, GY to every face next to liquid, except those touching
// the wall ring (after the copy the FLIP update compares against, so the
// particles feel it).
func (l *Liquid) gravity() {
	w, h := l.W, l.H
	uw := w + 1
	for j := 1; j < h-1; j++ {
		for i := 2; i < w-1; i++ {
			if c := j*w + i; l.wet(c-1) || l.wet(c) {
				l.u[j*uw+i] += l.GX
			}
		}
	}
	for j := 2; j < h-1; j++ {
		for i := 1; i < w-1; i++ {
			if c := j*w + i; l.wet(c-w) || l.wet(c) {
				l.v[c] += l.GY
			}
		}
	}
}

// Direction bits of a particle's last move.
const (
	movedR uint8 = 1 << iota
	movedL
	movedD
	movedU
)

// advect moves every particle by its velocity (limited to MaxV), except
// one slower than RestQ8/256 of what gravity adds per step. A still liquid
// has to stay exactly still; what the grid leaves under it (surface cells
// changing hands) is a fraction of gravity's pull, so the bar scales with
// it. A tilt leaves gravity unbalanced and moves the liquid within a few
// steps.
//
// A stopped particle keeps only HoldQ8/256 per step of a velocity that
// points back against its last move. A lone surface particle on a cell
// edge otherwise rocks for ever under steady gravity: it creeps across the
// edge, the cells change hands, the push reverses, its speed builds up
// past the bar while it stands, and it creeps back. Each time gravity has
// turned by more than 1/128 of its strength (about 0.45°) the last moves
// are forgotten, so tilting the other way is not damped. Smooth the
// accelerometer below that: noise above it keeps forgetting them (and
// moves the liquid by itself anyway).
func (l *Liquid) advect() {
	gx, gy := max(l.GX, -l.GX), max(l.GY, -l.GY)
	g := max(gx, gy) + min(gx, gy)*3/8 // |g| within 7 %, no square root
	if dx, dy := l.GX-l.lastGX, l.GY-l.lastGY; max(dx, -dx, dy, -dy) > g>>7 {
		clear(l.last)
		l.lastGX, l.lastGY = l.GX, l.GY
	}
	m, rest, hold := l.prm.MaxV, g*l.prm.RestQ8>>8, l.prm.HoldQ8
	for i := range l.X {
		vx, vy := clamp(vel(l.VX[i]), -m, m), clamp(vel(l.VY[i]), -m, m)
		if max(vx, -vx, vy, -vy) < rest {
			d := l.last[i]
			if (vx > 0 && d&movedL != 0) || (vx < 0 && d&movedR != 0) {
				vx = vx * hold / 256
			}
			if (vy > 0 && d&movedU != 0) || (vy < 0 && d&movedD != 0) {
				vy = vy * hold / 256
			}
			l.VX[i], l.VY[i] = store(vx), store(vy)
			continue
		}
		l.VX[i], l.VY[i] = store(vx), store(vy)
		// move by the velocity rounded to a position unit (1/256 cell)
		l.X[i] = uint16(int32(l.X[i]) + (vx+128)>>8)
		l.Y[i] = uint16(int32(l.Y[i]) + (vy+128)>>8)
		var d uint8
		if vx > 0 {
			d = movedR
		} else if vx < 0 {
			d = movedL
		}
		if vy > 0 {
			d |= movedD
		} else if vy < 0 {
			d |= movedU
		}
		l.last[i] = d
	}
}

// vel is a stored velocity (Q14) in the arithmetic's Q16.
func vel(v int16) int32 { return int32(v) << 2 }

// store is a Q16 velocity (|v| ≤ One) rounded to the stored Q14.
func store(v int32) int16 { return int16((v + 2) >> 2) }

func clamp(v, lo, hi int32) int32 { return min(max(v, lo), hi) }

// isqrt is ⌊√x⌋ for x < 2¹⁶ (digit by digit, no divide): a pair's d² is
// under (2 radii)², below 2¹⁵.
func isqrt(x int32) int32 {
	var r int32
	for b := int32(1) << 14; b != 0; b >>= 2 {
		if x >= r+b {
			x -= r + b
			r = r>>1 + b
		} else {
			r >>= 1
		}
	}
	return r
}

// walls keeps every particle inside the box, stopping its motion into the
// wall it touched.
func (l *Liquid) walls() {
	r := l.r
	lo, hiX, hiY := Cell+r, int32(l.W-1)*Cell-r, int32(l.H-1)*Cell-r
	for i := range l.X {
		if x := int32(l.X[i]); x < lo {
			l.X[i] = uint16(lo)
			l.VX[i] = max(l.VX[i], 0)
		} else if x > hiX {
			l.X[i] = uint16(hiX)
			l.VX[i] = min(l.VX[i], 0)
		}
		if y := int32(l.Y[i]); y < lo {
			l.Y[i] = uint16(lo)
			l.VY[i] = max(l.VY[i], 0)
		} else if y > hiY {
			l.Y[i] = uint16(hiY)
			l.VY[i] = min(l.VY[i], 0)
		}
	}
}

// bucket sorts the particles by cell (counting sort into order, with
// start[c]..start[c+1] the particles of cell c, start in aux) for
// separate.
func (l *Liquid) bucket() {
	n := l.W * l.H
	start := l.aux
	clear(start)
	for i := range l.X {
		start[l.cellOf(i)+1]++
	}
	for c := 1; c <= n; c++ {
		start[c] += start[c-1]
	}
	// fill using start[c] as a cursor, then shift it back
	for i := range l.X {
		c := l.cellOf(i)
		l.order[start[c]] = uint16(i)
		start[c]++
	}
	copy(start[1:], start[:n])
	start[0] = 0
}

// separate pushes apart every pair of particles closer than two radii d₀,
// each by half the overlap (d₀ − d) along the line between them. One cell
// is at least two radii, so a pair is in the same cell or in neighbouring
// ones: each cell is matched with itself and with the four neighbours
// after it (right, and the three below), which visits every pair once.
// The buckets are from the start of the pass sequence; the pushes are too
// small to matter in between.
func (l *Liquid) separate() {
	w, h := l.W, l.H
	start := l.aux
	for cy := range h {
		for cx := range w {
			c := cy*w + cx
			a0, a1 := start[c], start[c+1]
			for a := a0; a < a1; a++ {
				i := int(l.order[a])
				for b := a + 1; b < a1; b++ {
					l.pair(i, int(l.order[b]))
				}
				if cx+1 < w {
					l.pairs(i, c+1)
				}
				if cy+1 < h {
					if cx > 0 {
						l.pairs(i, c+w-1)
					}
					l.pairs(i, c+w)
					if cx+1 < w {
						l.pairs(i, c+w+1)
					}
				}
			}
		}
	}
}

// pairs separates particle i from every particle of cell c.
func (l *Liquid) pairs(i, c int) {
	for b := l.aux[c]; b < l.aux[c+1]; b++ {
		l.pair(i, int(l.order[b]))
	}
}

// pair pushes particles i and j apart if they overlap. Only overlapping
// pairs (none in a liquid packed at rest) cost a square root and a divide.
func (l *Liquid) pair(i, j int) {
	d0 := 2 * l.r
	dx, dy := int32(l.X[j])-int32(l.X[i]), int32(l.Y[j])-int32(l.Y[i])
	if dx >= d0 || dx <= -d0 || dy >= d0 || dy <= -d0 {
		return
	}
	d2 := dx*dx + dy*dy
	if d2 >= d0*d0 {
		return
	}
	var px, py int32
	if d2 == 0 { // on top of each other: split along x
		px = d0 / 2
	} else {
		d := isqrt(d2)
		px, py = half(dx*(d0-d), d), half(dy*(d0-d), d)
	}
	l.X[i] = uint16(int32(l.X[i]) - px)
	l.Y[i] = uint16(int32(l.Y[i]) - py)
	l.X[j] = uint16(int32(l.X[j]) + px)
	l.Y[j] = uint16(int32(l.Y[j]) + py)
}

// half is a/(2b) rounded to the nearest position unit (b > 0). Truncated,
// a push falls short of a whole-unit move by one unit, and a particle
// pressed into its neighbour wobbles by that unit for ever.
func half(a, b int32) int32 {
	if a < 0 {
		return -((-a + b) / (2 * b))
	}
	return (a + b) / (2 * b)
}

func (l *Liquid) cellOf(i int) int {
	cx := min(int(l.X[i]>>8), l.W-1)
	cy := min(int(l.Y[i]>>8), l.H-1)
	return cy*l.W + cx
}

// weights splits a position (1/256 cell) into its cell and the four
// bilinear weights (Q8, summing to 256).
func weights(x, y int32) (i, j int, w00, w10, w01, w11 int32) {
	i, j = int(x>>8), int(y>>8)
	fx, fy := x&255, y&255
	w00 = (256 - fx) * (256 - fy) >> 8
	w10 = fx * (256 - fy) >> 8
	w01 = (256 - fx) * fy >> 8
	w11 = fx * fy >> 8
	return
}

// toGrid moves the particles' velocities onto the faces, marks the cells
// and keeps a copy of the face velocities for the FLIP update. A cell is
// wall (the ring), liquid where a particle is, and stays liquid while a
// particle is within Stick of it: marking only the cell a particle's
// centre is in would flip a surface cell between liquid and air each time
// a particle sitting on its edge moves a hair, and every flip changes the
// pressure under it, so the surface would never settle.
func (l *Liquid) toGrid() {
	w, h := l.W, l.H
	clear(l.u)
	clear(l.v)
	clear(l.pu)
	clear(l.pv)
	for j := range h {
		for i := range w {
			k := j*w + i
			c := air
			if l.cell[k]&kind == fluid {
				c = wasFluid
			}
			if i == 0 || j == 0 || i == w-1 || j == h-1 {
				c |= solid
			}
			l.cell[k] = c
		}
	}
	stick := l.stick
	for p := range l.X {
		c := l.cellOf(p)
		l.mark(c)
		fx, fy := int32(l.X[p])&255, int32(l.Y[p])&255
		if fx < stick && l.cell[c-1]&wasFluid != 0 {
			l.mark(c - 1)
		} else if fx >= Cell-stick && l.cell[c+1]&wasFluid != 0 {
			l.mark(c + 1)
		}
		if fy < stick && l.cell[c-w]&wasFluid != 0 {
			l.mark(c - w)
		} else if fy >= Cell-stick && l.cell[c+w]&wasFluid != 0 {
			l.mark(c + w)
		}
	}
	const half = Cell / 2
	if l.drift != 0 {
		dens := l.aux[:w*h]
		clear(dens)
		for p := range l.X {
			i, j, a, b, c, d := weights(int32(l.X[p])-half, int32(l.Y[p])-half)
			k := j*w + i
			dens[k] += uint16(a)
			dens[k+1] += uint16(b)
			dens[k+w] += uint16(c)
			dens[k+w+1] += uint16(d)
		}
	}
	// walls keeps every particle a radius inside the ring, so these
	// stencils stay on the grid; pu and pv sum the weights for now (16 bits
	// hold 128 particles' full weight on one face, far beyond any packing
	// separate allows; a caller stacking more only gets clamped speeds)
	uw := w + 1
	for p := range l.X {
		x, y := int32(l.X[p]), int32(l.Y[p])
		// u faces sit at (i, j+½)
		i, j, a, b, c, d := weights(x, y-half)
		k := j*uw + i
		vx := vel(l.VX[p])
		l.u[k] += a * vx
		l.u[k+1] += b * vx
		l.u[k+uw] += c * vx
		l.u[k+uw+1] += d * vx
		l.pu[k] += int16(a)
		l.pu[k+1] += int16(b)
		l.pu[k+uw] += int16(c)
		l.pu[k+uw+1] += int16(d)
		// v faces sit at (i+½, j)
		i, j, a, b, c, d = weights(x-half, y)
		k = j*w + i
		vy := vel(l.VY[p])
		l.v[k] += a * vy
		l.v[k+1] += b * vy
		l.v[k+w] += c * vy
		l.v[k+w+1] += d * vy
		l.pv[k] += int16(a)
		l.pv[k+1] += int16(b)
		l.pv[k+w] += int16(c)
		l.pv[k+w+1] += int16(d)
	}
	// normalise; faces touching the wall ring carry no flow; then keep
	// the result for the FLIP update
	for j := range h {
		for i := 0; i <= w; i++ {
			k := j*uw + i
			if s := int32(l.pu[k]); s > 0 && i > 1 && i < w-1 && j > 0 && j < h-1 {
				l.u[k] /= s
			} else {
				l.u[k] = 0
			}
			l.pu[k] = store(l.u[k])
		}
	}
	for j := 0; j <= h; j++ {
		for i := range w {
			k := j*w + i
			if s := int32(l.pv[k]); s > 0 && j > 1 && j < h-1 && i > 0 && i < w-1 {
				l.v[k] /= s
			} else {
				l.v[k] = 0
			}
			l.pv[k] = store(l.v[k])
		}
	}
}

// mark makes cell k liquid unless it is wall.
func (l *Liquid) mark(k int) {
	if l.cell[k]&kind == air {
		l.cell[k] |= fluid
	}
}

// recipQ8[s] = 1/s in Q8 for the open-side count s of a cell.
var recipQ8 = [5]int32{0, 256, 128, 85, 64}

// solve makes the face velocities divergence-free in fluid cells (Gauss–
// Seidel with over-relaxation). It starts from last step's pressure (warm
// start): a liquid at rest is then already balanced against gravity, and
// a jammed corner, which twenty sweeps from nothing leave a little
// unbalanced, settles too. A cell that is no longer liquid forgets it. The
// pressure is kept in 16 bits, enough for g·depth up to half a cell per
// step (0.02 cell per step² under 25 cells of liquid); beyond, it
// saturates and the warm start merely helps less.
func (l *Liquid) solve() {
	w, h := l.W, l.H
	for j := 1; j < h-1; j++ {
		for i := 1; i < w-1; i++ {
			k := j*w + i
			if !l.wet(k) {
				l.pres[k] = 0
				continue
			}
			l.push(i, j, int32(l.pres[k]))
		}
	}
	for range l.prm.Iterations {
		for j := 1; j < h-1; j++ {
			for i := 1; i < w-1; i++ {
				k := j*w + i
				if !l.wet(k) {
					continue
				}
				ku := k + j // j·(w+1) + i
				div := l.u[ku+1] - l.u[ku] + l.v[k+w] - l.v[k]
				if l.drift != 0 {
					c := int32(l.aux[k]) - l.rest
					// pull a loose cell together only deep in the liquid (all four
					// neighbours liquid): a surface cell is loose by nature, and so
					// is one against a wall, which particles keep a radius from
					if c > 0 || l.wet(k-1) && l.wet(k+1) && l.wet(k-w) && l.wet(k+w) {
						div -= c * l.drift >> 8
					}
				}
				open := 4
				if i == 1 {
					open--
				}
				if i == w-2 {
					open--
				}
				if j == 1 {
					open--
				}
				if j == h-2 {
					open--
				}
				p := (-div * l.prm.OverQ8 >> 8) * recipQ8[open] >> 8
				l.pres[k] = int16(clamp(int32(l.pres[k])+p, -32767, 32767))
				l.push(i, j, p)
			}
		}
	}
}

// push applies a pressure impulse p in cell (i, j): out through every
// side that does not face the wall ring.
func (l *Liquid) push(i, j int, p int32) {
	w := l.W
	k := j*w + i
	ku := k + j // its left u face, j·(w+1) + i
	if i > 1 {
		l.u[ku] -= p
	}
	if i < w-2 {
		l.u[ku+1] += p
	}
	if j > 1 {
		l.v[k] -= p
	}
	if j < l.H-2 {
		l.v[k+w] += p
	}
}

// fromGrid gives each particle the grid's velocity at its position (PIC)
// blended with its own velocity plus the grid's change this step (FLIP).
// FLIP keeps the liquid lively but never loses its noise; a particle
// slower than Calm gets proportionally more PIC, so a liquid coming to
// rest damps out instead of circulating. Only faces with liquid on a side
// count; the particle's own cell is liquid, so its two faces always do.
// (walls keeps every particle a radius inside the ring, so the stencils
// stay on the grid and every cell looked at exists.)
func (l *Liquid) fromGrid() {
	w := l.W
	uw := w + 1
	const half = Cell / 2
	for p := range l.X {
		vx, vy := vel(l.VX[p]), vel(l.VY[p])
		f := l.prm.FlipQ8
		if l.calm > 0 {
			s := max(vx, -vx, vy, -vy)
			f = f * min(s*l.calm>>16, 256) >> 8
		}
		x, y := int32(l.X[p]), int32(l.Y[p])
		// u face (fi, fj) lies between cells fj·w+fi−1 and fj·w+fi
		i, j, a, b, c, d := weights(x, y-half)
		k, cl := j*uw+i, j*w+i-1
		w0, w1, w2 := l.wet(cl), l.wet(cl+1), l.wet(cl+2)
		w3, w4, w5 := l.wet(cl+w), l.wet(cl+w+1), l.wet(cl+w+2)
		l.VX[p] = store(blend(vx, f, [4]int{k, k + 1, k + uw, k + uw + 1}, [4]int32{a, b, c, d},
			[4]bool{w0 || w1, w1 || w2, w3 || w4, w4 || w5}, l.u, l.pu))
		// v face (fi, fj) lies between cells (fj−1)·w+fi and fj·w+fi
		i, j, a, b, c, d = weights(x-half, y)
		k, ca := j*w+i, (j-1)*w+i
		w0, w1 = l.wet(ca), l.wet(ca+1)
		w3, w4 = l.wet(ca+w), l.wet(ca+w+1)
		w2, w5 = l.wet(ca+2*w), l.wet(ca+2*w+1)
		l.VY[p] = store(blend(vy, f, [4]int{k, k + 1, k + w, k + w + 1}, [4]int32{a, b, c, d},
			[4]bool{w0 || w3, w1 || w4, w3 || w2, w4 || w5}, l.v, l.pv))
	}
}

// blend is one velocity component of fromGrid: the PIC value and the FLIP
// change, each a weighted mean over the valid faces of the stencil.
func blend(vp, f int32, ks [4]int, ws [4]int32, ok [4]bool, g []int32, pg []int16) int32 {
	var sw, pic, corr int32
	for n, k := range ks {
		if ws[n] == 0 || !ok[n] {
			continue
		}
		sw += ws[n]
		pic += ws[n] * g[k]
		corr += ws[n] * (g[k] - vel(pg[k]))
	}
	pic /= sw
	corr /= sw
	return clamp(((256-f)*pic+f*(vp+corr))>>8, -One, One)
}

// Raster counts the particles over a w×h raster covering the whole box
// (wall ring included), saturating at 255: dst[y*w+x] is how many
// particles sit in that pixel. It is how a display draws the liquid at any
// pixel size; dst must hold w*h bytes.
func (l *Liquid) Raster(dst []uint8, w, h int) {
	dst = dst[:w*h]
	clear(dst)
	// pixels per cell in Q8, so x·w/W needs no divide per particle:
	// (position, 1/256 cell)·(Q8 scale) >> 16 is a pixel
	sx, sy := int32(w<<8/l.W), int32(h<<8/l.H)
	for i := range l.X {
		px := int(int32(l.X[i]) * sx >> 16)
		py := int(int32(l.Y[i]) * sy >> 16)
		if px >= w || py >= h {
			continue
		}
		if k := py*w + px; dst[k] < 255 {
			dst[k]++
		}
	}
}
