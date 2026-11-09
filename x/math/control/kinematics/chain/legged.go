package chain

import (
	"fmt"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Frames: world and body are x forward, y left, z up. A hip frame has its
// origin at Mount.Hip and is the body frame turned by Mount.Yaw about z; it is
// the leg chain's base frame (coxa yaw axis = z, q = 0 points along +x).

// LegGeometry is a coxa–femur–tibia leg: link lengths in metres.
type LegGeometry struct {
	Coxa, Femur, Tibia float32
}

// DefaultJointMaxVel is the joint speed limit Hexapod and Quadruped declare:
// 300°/s, a hobby servo at 0.2 s/60° without load.
const DefaultJointMaxVel = 300 * math32.Pi / 180

// DefaultLegLimits returns the joint limits Hexapod and Quadruped use (ER-R2
// probe R, knee down only): coxa ±60°, femur −60…72°, tibia −149…−20°.
func DefaultLegLimits() [3][2]float32 {
	return [3][2]float32{{Deg(-60), Deg(60)}, {Deg(-60), Deg(72)}, {Deg(-149), Deg(-20)}}
}

// CoxaFemurTibia declares a leg: the coxa yaws about the hip z axis (α = π/2
// turns the next axes horizontal), femur and tibia pitch in the vertical plane,
// positive up, the tibia angle relative to the femur. lim[j] is {Min, Max} of
// joint j in radians. The chain matches Planar3, so Model is analytic; g needs
// Femur, Tibia > 0 and Coxa >= 0, otherwise Planar3, Model and NewLegSolver
// return kintypes.ErrInvalidConfig.
func CoxaFemurTibia(name string, g LegGeometry, lim [3][2]float32, servo [3]ServoMap) Chain {
	return Chain{Name: name, Joints: []Joint{
		{Name: name + ".coxa", A: g.Coxa, Alpha: math32.Pi / 2, Min: lim[0][0], Max: lim[0][1], Servo: servo[0]},
		{Name: name + ".femur", A: g.Femur, Min: lim[1][0], Max: lim[1][1], Servo: servo[1]},
		{Name: name + ".tibia", A: g.Tibia, Min: lim[2][0], Max: lim[2][1], Servo: servo[2]},
	}}
}

// Mount places a leg chain on the body: hip position in the body frame and the
// hip frame's yaw about the body z axis.
type Mount struct {
	Leg Chain
	Hip vec.Vector3D
	Yaw float32
}

// Body is a legged body: one Mount per leg and each leg's neutral foot
// position in the body frame. Leg order follows gait/scheduler: left front to
// back, then right front to back.
type Body struct {
	Mounts  []Mount
	Neutral []vec.Vector3D
}

// Clone returns a deep copy of b: mounts, their leg joints and neutral feet
// share no memory with b.
func (b Body) Clone() Body {
	c := Body{Neutral: append([]vec.Vector3D(nil), b.Neutral...)}
	if b.Mounts != nil {
		c.Mounts = make([]Mount, len(b.Mounts))
	}
	for i, m := range b.Mounts {
		m.Leg.Joints = append([]Joint(nil), m.Leg.Joints...)
		c.Mounts[i] = m
	}
	return c
}

// Pose is the body pose in the world: position (Z = body height above the
// world origin) and Z-Y-X Euler angles, R = Rz(Yaw)·Ry(Pitch)·Rx(Roll).
type Pose struct {
	X, Y, Z          float32
	Roll, Pitch, Yaw float32
}

func (p Pose) rotation() [3][3]float32 {
	cr, sr := math32.Cos(p.Roll), math32.Sin(p.Roll)
	cp, sp := math32.Cos(p.Pitch), math32.Sin(p.Pitch)
	cy, sy := math32.Cos(p.Yaw), math32.Sin(p.Yaw)
	return [3][3]float32{
		{cy * cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr},
		{sy * cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr},
		{-sp, cp * sr, cp * cr},
	}
}

// ToWorld maps a body-frame point to the world.
func (p Pose) ToWorld(b vec.Vector3D) vec.Vector3D {
	r := p.rotation()
	return vec.Vector3D{
		r[0][0]*b[0] + r[0][1]*b[1] + r[0][2]*b[2] + p.X,
		r[1][0]*b[0] + r[1][1]*b[1] + r[1][2]*b[2] + p.Y,
		r[2][0]*b[0] + r[2][1]*b[1] + r[2][2]*b[2] + p.Z,
	}
}

// ToBody maps a world point to the body frame (inverse of ToWorld).
func (p Pose) ToBody(w vec.Vector3D) vec.Vector3D {
	r := p.rotation()
	x, y, z := w[0]-p.X, w[1]-p.Y, w[2]-p.Z
	return vec.Vector3D{
		r[0][0]*x + r[1][0]*y + r[2][0]*z,
		r[0][1]*x + r[1][1]*y + r[2][1]*z,
		r[0][2]*x + r[1][2]*y + r[2][2]*z,
	}
}

// BodyToLeg maps a body-frame point to leg i's hip frame.
func (b Body) BodyToLeg(i int, p vec.Vector3D) vec.Vector3D {
	m := &b.Mounts[i]
	c, s := math32.Cos(m.Yaw), math32.Sin(m.Yaw)
	x, y := p[0]-m.Hip[0], p[1]-m.Hip[1]
	return vec.Vector3D{c*x + s*y, -s*x + c*y, p[2] - m.Hip[2]}
}

// LegToBody maps a point in leg i's hip frame to the body frame.
func (b Body) LegToBody(i int, p vec.Vector3D) vec.Vector3D {
	m := &b.Mounts[i]
	c, s := math32.Cos(m.Yaw), math32.Sin(m.Yaw)
	return vec.Vector3D{c*p[0] - s*p[1] + m.Hip[0], s*p[0] + c*p[1] + m.Hip[1], p[2] + m.Hip[2]}
}

// FootInLeg maps a world foot position to leg i's hip frame (world → body →
// hip), the target frame of the leg's Model.
func (b Body) FootInLeg(i int, pose Pose, footWorld vec.Vector3D) vec.Vector3D {
	return b.BodyToLeg(i, pose.ToBody(footWorld))
}

// FootInWorld maps a point in leg i's hip frame to the world (inverse of
// FootInLeg).
func (b Body) FootInWorld(i int, pose Pose, footLeg vec.Vector3D) vec.Vector3D {
	return pose.ToWorld(b.LegToBody(i, footLeg))
}

// Validate checks that every mount has a neutral foot and a valid leg chain.
// Errors wrap kintypes.ErrInvalidConfig.
func (b Body) Validate() error {
	if len(b.Mounts) == 0 || len(b.Mounts) != len(b.Neutral) {
		return fmt.Errorf("%w: chain: body needs one neutral foot per mount, got %d mounts, %d feet", kintypes.ErrInvalidConfig, len(b.Mounts), len(b.Neutral))
	}
	for i, m := range b.Mounts {
		if !finite(m.Hip[0], m.Hip[1], m.Hip[2], m.Yaw) || !finite(b.Neutral[i][0], b.Neutral[i][1], b.Neutral[i][2]) {
			return fmt.Errorf("%w: chain: mount %d: non-finite hip, yaw or neutral foot", kintypes.ErrInvalidConfig, i)
		}
		if err := m.Leg.Validate(); err != nil {
			return fmt.Errorf("chain: mount %d: %w", i, err)
		}
	}
	return nil
}

// Hexapod builds a six-legged body. front, middle and rear are the hip x
// coordinates (body frame, x forward, so rear is usually negative) and the
// hips sit at y = ±halfWidth, z = 0, in leg order L1 L2 L3 R1 R2 R3. Each hip
// points radially (Yaw = atan2(hip.y, hip.x)). Legs use DefaultLegLimits,
// DefaultJointMaxVel and servo channels 3·leg + joint with each joint's limit
// range centred on 0…180°; femur and tibia servos of left legs are mirrored
// (Sign −1), coxa servos are not. The neutral foot is the pose (0, 0, −90°):
// Coxa + Femur out along the hip yaw, Tibia below the hip. Replace Mounts
// entries to use calibrated servo maps.
func Hexapod(front, middle, rear, halfWidth float32, g LegGeometry) Body {
	names := [6]string{"L1", "L2", "L3", "R1", "R2", "R3"}
	xs := [3]float32{front, middle, rear}
	var hips [6]vec.Vector3D
	for i := range 3 {
		hips[i] = vec.Vector3D{xs[i], halfWidth, 0}
		hips[3+i] = vec.Vector3D{xs[i], -halfWidth, 0}
	}
	return legged(names[:], hips[:], g)
}

// Quadruped builds a four-legged body with hips at (±halfLength, ±halfWidth,
// 0) in leg order LF LH RF RH (gait/scheduler QuadLF…QuadRH). Everything else
// is as for Hexapod.
func Quadruped(halfLength, halfWidth float32, g LegGeometry) Body {
	names := [4]string{"LF", "LH", "RF", "RH"}
	hips := [4]vec.Vector3D{{halfLength, halfWidth, 0}, {-halfLength, halfWidth, 0}, {halfLength, -halfWidth, 0}, {-halfLength, -halfWidth, 0}}
	return legged(names[:], hips[:], g)
}

func legged(names []string, hips []vec.Vector3D, g LegGeometry) Body {
	lim := DefaultLegLimits()
	b := Body{Mounts: make([]Mount, len(hips)), Neutral: make([]vec.Vector3D, len(hips))}
	reach := g.Coxa + g.Femur
	for i, hip := range hips {
		left := hip[1] > 0
		var servo [3]ServoMap
		for j := range servo {
			sign := int8(1)
			if left && j > 0 {
				sign = -1
			}
			servo[j] = centredServo(3*i+j, sign, lim[j])
		}
		leg := CoxaFemurTibia(names[i], g, lim, servo)
		for j := range leg.Joints {
			leg.Joints[j].MaxVel = DefaultJointMaxVel
		}
		yaw := math32.Atan2(hip[1], hip[0])
		b.Mounts[i] = Mount{Leg: leg, Hip: hip, Yaw: yaw}
		b.Neutral[i] = vec.Vector3D{hip[0] + reach*math32.Cos(yaw), hip[1] + reach*math32.Sin(yaw), hip[2] - g.Tibia}
	}
	return b
}

// centredServo maps the middle of lim to 90° (Ratio 1).
func centredServo(channel int, sign int8, lim [2]float32) ServoMap {
	mid := (lim[0] + lim[1]) / 2
	return ServoMap{Channel: channel, ZeroDeg: 90 - float32(sign)*mid*radToDeg, Sign: sign, Ratio: 1}
}

// LegSolver solves every leg of a Body for world foot targets without
// allocating: one Model per leg (analytic for coxa–femur–tibia legs, DH-DLS
// otherwise) and preallocated matrices. It is not safe for concurrent use
// (SolveLegs writes shared scratch matrices).
type LegSolver struct {
	body   Body
	models []kintypes.Bidirectional
	dst    []mat.Matrix
	ctl    []mat.Matrix
	st     []mat.Matrix
	// Interface views of dst/ctl/st, converted once so calls do not allocate.
	dstI, ctlI, stI []mattype.Matrix
}

// NewLegSolver validates the body and builds one model per leg; opts tune the
// DH solver of non-leg-shaped chains. The solver keeps a deep copy of b, so
// later edits to b have no effect.
func (b Body) NewLegSolver(opts ...Option) (*LegSolver, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	b = b.Clone()
	o := defaultOptions(opts)
	n := len(b.Mounts)
	s := &LegSolver{
		body:   b,
		models: make([]kintypes.Bidirectional, n),
		dst:    make([]mat.Matrix, n),
		ctl:    make([]mat.Matrix, n),
		st:     make([]mat.Matrix, n),
		dstI:   make([]mattype.Matrix, n),
		ctlI:   make([]mattype.Matrix, n),
		stI:    make([]mattype.Matrix, n),
	}
	for i, m := range b.Mounts {
		model, rows, err := m.Leg.model(o)
		if err != nil {
			return nil, fmt.Errorf("chain: mount %d: %w", i, err)
		}
		dof := len(m.Leg.Joints)
		s.models[i] = model
		s.dst[i], s.ctl[i], s.st[i] = mat.New(rows, 1), mat.New(dof, 1), mat.New(dof, 1)
		s.dstI[i], s.ctlI[i], s.stI[i] = s.dst[i], s.ctl[i], s.st[i]
	}
	return s, nil
}

// Body returns a deep copy of the body the solver was built for (allocates).
func (s *LegSolver) Body() Body { return s.body.Clone() }

// SolveLegs maps each world foot target to its hip frame (FootInLeg) and runs
// the leg's inverse kinematics. q[i] (one value per joint of leg i) seeds an
// iterative solver on input and receives the solution, or the solver's
// best-effort answer, on output; errs[i] receives leg i's typed error
// (kintypes.ErrUnreachable, ErrJointLimit, ErrNoConvergence) or nil. It
// returns kintypes.ErrInvalidDimensions for mis-sized arguments, otherwise the
// first leg error or nil. No allocation.
func (s *LegSolver) SolveLegs(pose Pose, feet []vec.Vector3D, q [][]float32, errs []error) error {
	n := len(s.models)
	if len(feet) != n || len(q) != n || len(errs) != n {
		return kintypes.ErrInvalidDimensions
	}
	for i := range n {
		if len(q[i]) != len(s.st[i]) {
			return kintypes.ErrInvalidDimensions
		}
	}
	var first error
	for i := range n {
		p := s.body.FootInLeg(i, pose, feet[i])
		dst, st, ctl := s.dst[i], s.st[i], s.ctl[i]
		dst[0][0], dst[1][0], dst[2][0] = p[0], p[1], p[2]
		for j, v := range q[i] {
			st[j][0] = v
		}
		err := s.models[i].Backward(s.stI[i], s.dstI[i], s.ctlI[i])
		for j := range q[i] {
			q[i][j] = ctl[j][0]
		}
		errs[i] = err
		if first == nil {
			first = err
		}
	}
	return first
}
