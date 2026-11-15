// Package boids is Reynolds flocking (separation, alignment, cohesion) in
// fixed point, inside a box that turns and shakes in the boids' world: the
// box is a window onto free space, so moving its walls sweeps them through
// the flock while boids away from the walls keep their world heading.
//
// Positions are in 1/Fix of a cell, velocities in 1/Fix of a cell per step.
// Neighbours are found by a plain scan with an early reject on |dx| and
// |dy|, O(N²): fine for the N ≤ 128 a badge runs. Speeds are kept between
// VMin and VMax measured with the octagonal norm max + 3/8·min (no square
// root). All steering divides rather than shifts: a shift rounds negative
// values towards −∞ and drifts the whole flock up and left.
//
// Integer only, deterministic on every target, no allocation after New.
// Ported from a Tufty 2040 badge demo (its boids and boids-box apps); the
// drawing stays in the demo.
package boids

import "errors"

// Fix is the number of position units per cell.
const Fix = 256

// TurnStep turns the box by 22.5° (256 = one turn), one button press.
const TurnStep = 16

// ShakeLen is the number of steps a shake takes (out and back).
const ShakeLen = 20

const q14 = 1 << 14

// ErrParams reports parameters New cannot run.
var ErrParams = errors.New("boids: invalid parameters")

// Params are the flocking rules, in position units (1/Fix cell).
type Params struct {
	R      int32 // neighbour radius
	Sep    int32 // closer than this, a neighbour is pushed away
	Margin int32 // inside this margin a wall pushes the boid inwards
	Push   int32 // velocity a wall adds per step inside the margin
	VMin   int32 // speed limits, octagonal norm, per step
	VMax   int32
}

// DefaultParams are the badge demo's: radius 6 cells, separation 2 cells,
// wall margin 4 cells, speeds 56…128 per step (0.22…0.5 cell).
func DefaultParams() Params {
	return Params{R: 6 * Fix, Sep: 2 * Fix, Margin: 4 * Fix, Push: 16, VMin: 56, VMax: 128}
}

// Box is the window: W×H position units centred at (CX, CY) in the world,
// turned by Angle (256 per turn). Angle steps one unit per Advance towards
// Target; Spin keeps raising Target.
type Box struct {
	W, H          int32
	Angle, Target int
	Spin          bool
	BaseX, BaseY  int32 // centre without the shake
	CX, CY        int32 // centre now

	shakeT           int
	shakeDX, shakeDY int32
	frame            int
}

// Flock is up to len(X) boids, the first N of them live; X, Y, VX, VY are
// in world coordinates.
type Flock struct {
	Params
	Box
	X, Y, VX, VY []int32
	N            int

	nvx, nvy []int32
	r2, sep2 int32 // radius and separation squared, in (1/16 cell)²
	rng      uint32
}

// New makes a flock of n boids in a w×h-cell box (w, h ≤ 256) centred in
// the world, seeded with Seed(1). It is the only call that allocates.
func New(n int, w, h int32, p Params) (*Flock, error) {
	if n < 1 || w < 1 || h < 1 || w > 256 || h > 256 ||
		p.VMin < 1 || p.VMax < p.VMin || p.Sep < 0 || p.R < p.Sep || p.R > 32*Fix || p.Margin < 0 || p.Push < 0 {
		return nil, ErrParams
	}
	buf := make([]int32, 6*n)
	f := &Flock{
		Params: p,
		X:      buf[0*n : 1*n], Y: buf[1*n : 2*n], VX: buf[2*n : 3*n], VY: buf[3*n : 4*n],
		nvx: buf[4*n : 5*n], nvy: buf[5*n : 6*n],
		N:    n,
		r2:   (p.R / 16) * (p.R / 16),
		sep2: (p.Sep / 16) * (p.Sep / 16),
	}
	f.Box = Box{W: w * Fix, H: h * Fix, BaseX: w * Fix / 2, BaseY: h * Fix / 2}
	f.Box.CX, f.Box.CY = f.Box.BaseX, f.Box.BaseY
	f.Seed(1)
	return f, nil
}

// Seed scatters every boid (all len(X)) at a random place in the box and a
// random velocity, from a xorshift32 seeded with seed (0 counts as 1).
func (f *Flock) Seed(seed uint32) {
	if seed == 0 {
		seed = 1
	}
	f.rng = seed
	for i := range f.X {
		bx := int32(f.rand() % uint32(f.Box.W))
		by := int32(f.rand() % uint32(f.Box.H))
		f.X[i], f.Y[i] = f.Box.ToWorld(bx, by)
		f.VX[i] = int32(f.rand()%uint32(2*f.VMax)) - f.VMax
		f.VY[i] = int32(f.rand()%uint32(2*f.VMax)) - f.VMax
	}
}

func (f *Flock) rand() uint32 {
	f.rng ^= f.rng << 13
	f.rng ^= f.rng >> 17
	f.rng ^= f.rng << 5
	return f.rng
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

// limit keeps the speed between VMin and VMax. A velocity that steering
// cancelled to zero keeps the old one (ox, oy): falling back to a fixed
// direction would make every stalled boid turn the same way, a bias the
// mirror test catches.
func (f *Flock) limit(vx, vy, ox, oy int32) (int32, int32) {
	if vx == 0 && vy == 0 {
		vx, vy = ox, oy
	}
	ax, ay := abs32(vx), abs32(vy)
	mag := max(ax, ay) + min(ax, ay)>>2 + min(ax, ay)>>3
	switch {
	case mag > f.VMax:
		return vx * f.VMax / mag, vy * f.VMax / mag
	case mag < f.VMin && mag > 0:
		return vx * f.VMin / mag, vy * f.VMin / mag
	case mag == 0:
		return f.VMin, 0 // no heading at all (only a seeded zero velocity)
	}
	return vx, vy
}

// Step moves the flock one step: every boid steers from its neighbours as
// they were at the start of the step, then all move, then the box's walls
// act on the boids they reach.
func (f *Flock) Step() {
	n := f.N
	R := f.R
	for i := range n {
		xi, yi := f.X[i], f.Y[i]
		var cnt, sx, sy, svx, svy, px, py int32
		for j := range n {
			if j == i {
				continue
			}
			dx, dy := f.X[j]-xi, f.Y[j]-yi
			if dx > R || dx < -R || dy > R || dy < -R {
				continue
			}
			// /16, not >>4: in (1/16 cell)² the squares fit int32 for any
			// box; the division keeps negative offsets as near as positive
			d2 := (dx/16)*(dx/16) + (dy/16)*(dy/16)
			if d2 > f.r2 {
				continue
			}
			cnt++
			sx += dx
			sy += dy
			svx += f.VX[j]
			svy += f.VY[j]
			if d2 < f.sep2 {
				px -= dx
				py -= dy
			}
		}
		vx, vy := f.VX[i], f.VY[i]
		if cnt > 0 {
			vx += (svx/cnt - vx) / 8 // alignment
			vy += (svy/cnt - vy) / 8
			vx += sx / cnt / 64 // cohesion: towards the neighbours' centre
			vy += sy / cnt / 64
		}
		vx += px / 8 // separation
		vy += py / 8
		f.nvx[i], f.nvy[i] = f.limit(vx, vy, f.VX[i], f.VY[i])
	}
	for i := range n {
		f.VX[i], f.VY[i] = f.nvx[i], f.nvy[i]
		f.X[i] += f.VX[i]
		f.Y[i] += f.VY[i]
		f.walls(i)
	}
}

// walls applies the box's walls to boid i. Seen from the box, a boid inside
// the margin is pushed inwards and one past a wall (the wall moved, or it
// flew) is put back and bounces; the minimum speed would otherwise undo
// the push for a boid flying straight at a wall. Only boids a wall reaches
// are converted to box coordinates and back, so the others keep their
// world position and heading exactly.
func (f *Flock) walls(i int) {
	b := &f.Box
	hiX, hiY, m := b.W-1, b.H-1, f.Margin
	bx, by := b.ToBox(f.X[i], f.Y[i])
	if bx >= m && bx <= hiX-m && by >= m && by <= hiY-m {
		return
	}
	vx, vy := b.VecToBox(f.VX[i], f.VY[i])
	switch {
	case bx < m:
		vx += f.Push
	case bx > hiX-m:
		vx -= f.Push
	}
	switch {
	case by < m:
		vy += f.Push
	case by > hiY-m:
		vy -= f.Push
	}
	nx, nvx := bounce(bx, vx, hiX)
	ny, nvy := bounce(by, vy, hiY)
	nx, ny = min(max(nx, 0), hiX), min(max(ny, 0), hiY) // a fast wall can throw it far
	if nx != bx || ny != by {
		f.X[i], f.Y[i] = b.ToWorld(nx, ny)
	}
	wvx, wvy := b.VecToWorld(nvx, nvy)
	f.VX[i], f.VY[i] = f.limit(wvx, wvy, f.VX[i], f.VY[i])
}

func bounce(p, v, hi int32) (int32, int32) {
	switch {
	case p < 0:
		return -p, abs32(v)
	case p > hi:
		return 2*hi - p, -abs32(v)
	}
	return p, v
}

// Scatter sends every live boid away from the box's centre (at least at
// VMin, at most at VMax).
func (f *Flock) Scatter() {
	for i := range f.N {
		f.VX[i], f.VY[i] = f.limit(f.X[i]-f.Box.CX, f.Y[i]-f.Box.CY, f.VX[i], f.VY[i])
	}
}

// Cell gives boid i's cell in the box, clamped to the box, and its heading
// octant on screen (see Octant), for drawing.
func (f *Flock) Cell(i int) (x, y int, octant uint8) {
	bx, by := f.Box.ToBox(f.X[i], f.Y[i])
	bx, by = min(max(bx, 0), f.Box.W-1), min(max(by, 0), f.Box.H-1)
	return int(bx / Fix), int(by / Fix), Octant(f.Box.VecToBox(f.VX[i], f.VY[i]))
}

// Octant is the heading of (vx, vy): 0 east, counting clockwise on a
// screen with y down (2 south, 4 west, 6 north).
func Octant(vx, vy int32) uint8 {
	ax, ay := abs32(vx), abs32(vy)
	// steep, shallow or diagonal by tan(22.5°) ≈ 106/256
	var o uint8
	switch {
	case ay*256 < ax*106:
		o = 0
	case ax*256 < ay*106:
		o = 2
	default:
		o = 1
	}
	switch {
	case vx >= 0 && vy >= 0:
		return o // 0 E, 1 SE, 2 S
	case vx < 0 && vy >= 0:
		return 4 - o // 4 W, 3 SW, 2 S
	case vx < 0 && vy < 0:
		return (4 + o) & 7 // 4 W, 5 NW, 6 N
	default:
		return (8 - o) & 7 // 0 E, 7 NE, 6 N
	}
}

// Advance moves the box one step: Angle one unit towards Target (Spin
// raises Target every second step) and along a shake.
func (b *Box) Advance() {
	b.frame++
	if b.Spin && b.frame&1 == 0 {
		b.Target++
	}
	switch {
	case b.Angle < b.Target:
		b.Angle++
	case b.Angle > b.Target:
		b.Angle--
	}
	b.CX, b.CY = b.BaseX, b.BaseY
	if b.shakeT > 0 {
		// a triangle: out to the full throw at the middle, back at the end
		k := int32(ShakeLen/2 - abs(b.shakeT-ShakeLen/2))
		b.CX += b.shakeDX * k / (ShakeLen / 2)
		b.CY += b.shakeDY * k / (ShakeLen / 2)
		if b.shakeT++; b.shakeT > ShakeLen {
			b.shakeT = 0
		}
	}
}

// Shake throws the box dist position units in direction dir (256 per
// turn, 0 = +x) and back over ShakeLen steps.
func (b *Box) Shake(dir uint8, dist int32) {
	b.shakeDX = sinQ14[uint8(dir+64)] * dist / q14
	b.shakeDY = sinQ14[dir] * dist / q14
	b.shakeT = 1
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func (b *Box) sincos() (int32, int32) {
	a := uint8(b.Angle)
	return sinQ14[a], sinQ14[uint8(a+64)]
}

// ToBox maps a world point into box coordinates (0…W across).
func (b *Box) ToBox(x, y int32) (int32, int32) {
	s, c := b.sincos()
	dx, dy := x-b.CX, y-b.CY
	return (c*dx+s*dy)/q14 + b.W/2, (-s*dx+c*dy)/q14 + b.H/2
}

// ToWorld maps box coordinates back to the world.
func (b *Box) ToWorld(bx, by int32) (int32, int32) {
	s, c := b.sincos()
	dx, dy := bx-b.W/2, by-b.H/2
	return (c*dx-s*dy)/q14 + b.CX, (s*dx+c*dy)/q14 + b.CY
}

// VecToBox rotates a world velocity into the box's frame.
func (b *Box) VecToBox(vx, vy int32) (int32, int32) {
	s, c := b.sincos()
	return (c*vx + s*vy) / q14, (-s*vx + c*vy) / q14
}

// VecToWorld rotates a box-frame velocity into the world.
func (b *Box) VecToWorld(vx, vy int32) (int32, int32) {
	s, c := b.sincos()
	return (c*vx - s*vy) / q14, (s*vx + c*vy) / q14
}

// sinQ14 is sin(2πi/256) in Q14, rounded half away from zero.
var sinQ14 = [256]int32{
	0, 402, 804, 1205, 1606, 2006, 2404, 2801, 3196, 3590, 3981, 4370, 4756, 5139, 5520, 5897,
	6270, 6639, 7005, 7366, 7723, 8076, 8423, 8765, 9102, 9434, 9760, 10080, 10394, 10702, 11003, 11297,
	11585, 11866, 12140, 12406, 12665, 12916, 13160, 13395, 13623, 13842, 14053, 14256, 14449, 14635, 14811, 14978,
	15137, 15286, 15426, 15557, 15679, 15791, 15893, 15986, 16069, 16143, 16207, 16261, 16305, 16340, 16364, 16379,
	16384, 16379, 16364, 16340, 16305, 16261, 16207, 16143, 16069, 15986, 15893, 15791, 15679, 15557, 15426, 15286,
	15137, 14978, 14811, 14635, 14449, 14256, 14053, 13842, 13623, 13395, 13160, 12916, 12665, 12406, 12140, 11866,
	11585, 11297, 11003, 10702, 10394, 10080, 9760, 9434, 9102, 8765, 8423, 8076, 7723, 7366, 7005, 6639,
	6270, 5897, 5520, 5139, 4756, 4370, 3981, 3590, 3196, 2801, 2404, 2006, 1606, 1205, 804, 402,
	0, -402, -804, -1205, -1606, -2006, -2404, -2801, -3196, -3590, -3981, -4370, -4756, -5139, -5520, -5897,
	-6270, -6639, -7005, -7366, -7723, -8076, -8423, -8765, -9102, -9434, -9760, -10080, -10394, -10702, -11003, -11297,
	-11585, -11866, -12140, -12406, -12665, -12916, -13160, -13395, -13623, -13842, -14053, -14256, -14449, -14635, -14811, -14978,
	-15137, -15286, -15426, -15557, -15679, -15791, -15893, -15986, -16069, -16143, -16207, -16261, -16305, -16340, -16364, -16379,
	-16384, -16379, -16364, -16340, -16305, -16261, -16207, -16143, -16069, -15986, -15893, -15791, -15679, -15557, -15426, -15286,
	-15137, -14978, -14811, -14635, -14449, -14256, -14053, -13842, -13623, -13395, -13160, -12916, -12665, -12406, -12140, -11866,
	-11585, -11297, -11003, -10702, -10394, -10080, -9760, -9434, -9102, -8765, -8423, -8076, -7723, -7366, -7005, -6639,
	-6270, -5897, -5520, -5139, -4756, -4370, -3981, -3590, -3196, -2801, -2404, -2006, -1606, -1205, -804, -402,
}
