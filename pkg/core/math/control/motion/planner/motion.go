package planner

import (
	"fmt"
	"math"

	"github.com/chewxy/math32"
	vaj "github.com/itohio/EasyRobot/pkg/core/math/control/motion"
	"github.com/itohio/EasyRobot/pkg/core/math/control/pid"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const (
	epsilonDistance = 1e-4
	pidLimit        = float32(math.MaxFloat32)
)

// Motion maintains stateful trajectory planning utilities.
type Motion struct {
	constraints Constraints
	params      Parameters

	speedProfile vaj.VAJ1D
	speedPID     pid.PID1D
	headingPID   pid.PID1D
	lateralPID   pid.PID1D

	progress      float32
	pathSignature uint64
	lastYawRate   float32
	lastControls  Controls
	lastState     State
	initialized   bool
}

// NewMotion creates a planner respecting supplied constraints and parameters.
func NewMotion(constraints Constraints, params Parameters) (*Motion, error) {
	if err := validateConstraints(constraints); err != nil {
		return nil, fmt.Errorf("%w: constraints", err)
	}
	if err := validateParameters(params); err != nil {
		return nil, fmt.Errorf("%w: parameters", err)
	}

	m := &Motion{
		constraints: constraints,
		params:      params,
		speedProfile: vaj.New1D(
			constraints.MaxSpeed,
			constraints.MaxAcceleration,
			constraints.MaxJerk,
		),
		speedPID:   pid.New1D(params.SpeedPID.P, params.SpeedPID.I, params.SpeedPID.D, -pidLimit, pidLimit),
		headingPID: pid.New1D(params.HeadingPID.P, params.HeadingPID.I, params.HeadingPID.D, -pidLimit, pidLimit),
		lateralPID: pid.New1D(params.LateralPID.P, params.LateralPID.I, params.LateralPID.D, -pidLimit, pidLimit),
	}
	m.speedPID.Reset()
	m.headingPID.Reset()
	m.lateralPID.Reset()

	return m, nil
}

// Forward plans the next trajectory segment given current state and controls.
func (m *Motion) Forward(state State, controls Controls, path []vec.Vector3D) (Trajectory, error) {
	ctx, err := newPathContext(path)
	if err != nil {
		return Trajectory{}, err
	}
	m.handlePathChange(ctx.signature)

	m.progress = m.projectProgress(ctx, state.Position)
	target := clampFloat(m.progress+m.params.LookaheadDistance, 0, ctx.total)

	profile := m.advanceProfile(target, ctx.total)
	pos, tangent := ctx.sample(profile.progress)
	yaw, yawRate := m.calculateYaw(state.Yaw, tangent)

	curvature := ctx.curvature(profile.progress)
	remaining := ctx.total - profile.progress
	speed := m.limitSpeed(profile.velocity, curvature, remaining)

	next := State{
		Position:  pos,
		Yaw:       yaw,
		Speed:     speed,
		Timestamp: state.Timestamp + m.params.SamplePeriod,
	}
	nextControls := Controls{
		Linear:  speed,
		Angular: yawRate,
		Effort:  controls.Effort,
	}

	m.lastState = next
	m.lastControls = nextControls
	m.initialized = true

	return Trajectory{
		States:   []State{next},
		Controls: []Controls{nextControls},
	}, nil
}

// Backward computes control commands to follow the desired trajectory.
func (m *Motion) Backward(tr Trajectory, state State, controls Controls) (Controls, error) {
	if len(tr.States) == 0 {
		return Controls{}, ErrInvalidTrajectory
	}
	if !m.initialized {
		m.lastControls = controls
		m.lastState = state
		m.initialized = true
	}

	desired := tr.States[0]
	dt := m.params.SamplePeriod

	m.speedPID.Input = state.Speed
	m.speedPID.Target = desired.Speed
	m.speedPID.Update(dt)

	m.headingPID.Input = normalizeAngle(state.Yaw)
	m.headingPID.Target = normalizeAngle(desired.Yaw)
	m.headingPID.Update(dt)

	lateralErr := lateralError(state.Position, desired.Position, desired.Yaw)
	m.lateralPID.Input = lateralErr
	m.lateralPID.Target = 0
	m.lateralPID.Update(dt)

	linear := controls.Linear + m.speedPID.Output
	linear = clampFloat(linear, controls.Linear-m.constraints.MaxDeceleration*dt, controls.Linear+m.constraints.MaxAcceleration*dt)
	linear = clampFloat(linear, -m.constraints.MaxSpeed, m.constraints.MaxSpeed)

	angular := controls.Angular + m.headingPID.Output + m.lateralPID.Output
	angular = clampFloat(angular, -m.constraints.MaxTurnRate, m.constraints.MaxTurnRate)
	angular = m.applyAngularAcceleration(angular, dt, controls.Angular)

	if m.constraints.MaxLateralAcceleration > 0 {
		angular = limitForTorque(linear, angular, m.constraints.MaxLateralAcceleration)
	}

	cmd := Controls{
		Linear:  linear,
		Angular: angular,
		Effort:  controls.Effort,
	}
	m.lastControls = cmd

	return cmd, nil
}

func (m *Motion) handlePathChange(signature uint64) {
	if m.pathSignature == signature {
		return
	}
	m.pathSignature = signature
	m.progress = 0
	m.speedProfile.Reset()
	m.lastYawRate = 0
	m.initialized = false
}

func (m *Motion) projectProgress(ctx pathContext, pos vec.Vector3D) float32 {
	if ctx.total <= 0 {
		return 0
	}
	return projectOntoPath(pos, ctx)
}

func (m *Motion) advanceProfile(target, total float32) profileSnapshot {
	if target > total {
		target = total
	}
	delta := target - m.progress
	if delta <= 0 {
		return profileSnapshot{
			progress: m.progress,
			velocity: math32.Abs(m.speedProfile.Velocity),
		}
	}
	if delta < 1 {
		velocity := clampFloat(delta/m.params.SamplePeriod, 0, m.constraints.MaxSpeed)
		m.progress = target
		m.speedProfile.Input = target
		m.speedProfile.Output = target
		m.speedProfile.Velocity = velocity
		return profileSnapshot{
			progress: target,
			velocity: velocity,
		}
	}
	m.speedProfile.Input = m.progress
	m.speedProfile.Target = target
	m.speedProfile.Update(m.params.SamplePeriod)

	progress := clampFloat(m.speedProfile.Output, 0, total)
	m.progress = progress

	return profileSnapshot{
		progress: progress,
		velocity: math32.Abs(m.speedProfile.Velocity),
	}
}

func (m *Motion) calculateYaw(currentYaw float32, tangent vec.Vector3D) (float32, float32) {
	targetYaw := math32.Atan2(tangent[1], tangent[0])
	targetYaw = normalizeAngle(targetYaw)

	dt := m.params.SamplePeriod
	maxDelta := m.constraints.MaxTurnRate * dt
	delta := clampFloat(normalizeAngle(targetYaw-currentYaw), -maxDelta, maxDelta)

	yawRate := delta / dt
	maxAccel := m.constraints.MaxTurnAcceleration
	if maxAccel > 0 {
		yawAccel := (yawRate - m.lastYawRate) / dt
		yawAccel = clampFloat(yawAccel, -maxAccel, maxAccel)
		yawRate = m.lastYawRate + yawAccel*dt
		delta = yawRate * dt
	}
	m.lastYawRate = yawRate

	nextYaw := normalizeAngle(currentYaw + delta)
	return nextYaw, yawRate
}

func (m *Motion) limitSpeed(velocity, curvature, remaining float32) float32 {
	speed := clampFloat(velocity, 0, m.constraints.MaxSpeed)
	if m.constraints.MaxLateralAcceleration > 0 && math32.Abs(curvature) > epsilonDistance {
		latLimited := math32.Sqrt(m.constraints.MaxLateralAcceleration / math32.Abs(curvature))
		if latLimited < speed {
			speed = latLimited
		}
	}
	if m.constraints.MaxDeceleration > 0 {
		stop := (speed * speed) / (2 * m.constraints.MaxDeceleration)
		if stop > remaining {
			clamped := math32.Sqrt(2 * m.constraints.MaxDeceleration * math32.Max(remaining, 0))
			if clamped < speed {
				speed = clamped
			}
		}
	}
	return speed
}

func (m *Motion) applyAngularAcceleration(angular, dt, prev float32) float32 {
	maxChange := m.constraints.MaxTurnAcceleration * dt
	return clampFloat(angular, prev-maxChange, prev+maxChange)
}

type profileSnapshot struct {
	progress float32
	velocity float32
}

type pathContext struct {
	points    []vec.Vector3D
	lengths   []float32
	total     float32
	signature uint64
}

func newPathContext(path []vec.Vector3D) (pathContext, error) {
	if len(path) < 2 {
		return pathContext{}, ErrInvalidPath
	}
	lengths := computeArcLengths(path)
	total := lengths[len(lengths)-1]
	if total <= 0 {
		return pathContext{}, ErrInvalidPath
	}
	return pathContext{
		points:    path,
		lengths:   lengths,
		total:     total,
		signature: hashPath(path),
	}, nil
}

func (p pathContext) sample(progress float32) (vec.Vector3D, vec.Vector3D) {
	if progress <= 0 {
		return p.points[0], unitTangent(p.points[0], p.points[1])
	}
	if progress >= p.total {
		n := len(p.points)
		return p.points[n-1], unitTangent(p.points[n-2], p.points[n-1])
	}
	idx := searchSegment(p.lengths, progress)
	start := p.points[idx]
	end := p.points[idx+1]
	segLen := p.lengths[idx+1] - p.lengths[idx]
	ratio := float32(0)
	if segLen > 0 {
		ratio = (progress - p.lengths[idx]) / segLen
	}
	pos := interpolate(start, end, ratio)
	tangent := unitTangent(start, end)
	return pos, tangent
}

func (p pathContext) curvature(progress float32) float32 {
	if len(p.points) < 3 {
		return 0
	}
	delta := clampFloat(progress, 0, p.total)
	prev := clampFloat(delta-math32.Max(p.total*0.01, 0.05), 0, p.total)
	next := clampFloat(delta+math32.Max(p.total*0.01, 0.05), 0, p.total)
	p0, _ := p.sample(prev)
	p1, _ := p.sample(delta)
	p2, _ := p.sample(next)
	return curvature2D(p0, p1, p2)
}

func computeArcLengths(path []vec.Vector3D) []float32 {
	lengths := make([]float32, len(path))
	var total float32
	for i := 1; i < len(path); i++ {
		d := distance(path[i-1], path[i])
		total += d
		lengths[i] = total
	}
	return lengths
}

func projectOntoPath(pos vec.Vector3D, ctx pathContext) float32 {
	best := float32(0)
	minDist := float32(math32.MaxFloat32)
	for i := 0; i < len(ctx.points)-1; i++ {
		a := ctx.points[i]
		b := ctx.points[i+1]
		proj := projectPoint(pos, a, b)
		d := distance(pos, proj)
		if d < minDist {
			minDist = d
			best = ctx.lengths[i] + distance(a, proj)
		}
	}
	return clampFloat(best, 0, ctx.total)
}

func projectPoint(p, a, b vec.Vector3D) vec.Vector3D {
	ab := subtract(b, a)
	ap := subtract(p, a)
	len2 := dot(ab, ab)
	if len2 <= epsilonDistance {
		return a
	}
	t := clampFloat(dot(ap, ab)/len2, 0, 1)
	return interpolate(a, b, t)
}

func interpolate(a, b vec.Vector3D, t float32) vec.Vector3D {
	return vec.Vector3D{
		a[0] + (b[0]-a[0])*t,
		a[1] + (b[1]-a[1])*t,
		a[2] + (b[2]-a[2])*t,
	}
}

func unitTangent(a, b vec.Vector3D) vec.Vector3D {
	diff := subtract(b, a)
	length := math32.Sqrt(dot(diff, diff))
	if length <= epsilonDistance {
		return vec.Vector3D{1, 0, 0}
	}
	return vec.Vector3D{
		diff[0] / length,
		diff[1] / length,
		diff[2] / length,
	}
}

func distance(a, b vec.Vector3D) float32 {
	diff := subtract(b, a)
	return math32.Sqrt(dot(diff, diff))
}

func subtract(a, b vec.Vector3D) vec.Vector3D {
	return vec.Vector3D{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
	}
}

func dot(a, b vec.Vector3D) float32 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func curvature2D(p0, p1, p2 vec.Vector3D) float32 {
	x1, y1 := p0[0], p0[1]
	x2, y2 := p1[0], p1[1]
	x3, y3 := p2[0], p2[1]

	den := (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
	if math32.Abs(den) <= epsilonDistance {
		return 0
	}

	a := distance(p0, p1)
	b := distance(p1, p2)
	c := distance(p2, p0)
	if a <= epsilonDistance || b <= epsilonDistance || c <= epsilonDistance {
		return 0
	}

	area2 := math32.Abs(den)
	return (2 * area2) / (a * b * c)
}

func searchSegment(lengths []float32, progress float32) int {
	lo := 0
	hi := len(lengths) - 2
	for lo <= hi {
		mid := (lo + hi) / 2
		if progress < lengths[mid] {
			hi = mid - 1
			continue
		}
		if progress > lengths[mid+1] {
			lo = mid + 1
			continue
		}
		return mid
	}
	if lo < 0 {
		return 0
	}
	if lo > len(lengths)-2 {
		return len(lengths) - 2
	}
	return lo
}

func hashPath(path []vec.Vector3D) uint64 {
	var h uint64 = 1469598103934665603
	for _, p := range path {
		h = (h ^ uint64(math32.Float32bits(p[0]))) * 1099511628211
		h = (h ^ uint64(math32.Float32bits(p[1]))) * 1099511628211
		h = (h ^ uint64(math32.Float32bits(p[2]))) * 1099511628211
	}
	return h
}

func normalizeAngle(angle float32) float32 {
	for angle > math32.Pi {
		angle -= 2 * math32.Pi
	}
	for angle < -math32.Pi {
		angle += 2 * math32.Pi
	}
	return angle
}

func lateralError(actual, desired vec.Vector3D, heading float32) float32 {
	dx := desired[0] - actual[0]
	dy := desired[1] - actual[1]
	nx := -math32.Sin(heading)
	ny := math32.Cos(heading)
	return dx*nx + dy*ny
}

func limitForTorque(linear, angular, maxLat float32) float32 {
	// Lateral acceleration approx = v^2 * curvature = v^2 * angular / v = v * angular (assuming angular = v * curvature)
	// Clamp angular such that |linear * angular| <= maxLat
	if maxLat <= 0 {
		return angular
	}
	if math32.Abs(linear) <= epsilonDistance {
		return angular
	}
	maxAngular := maxLat / math32.Max(math32.Abs(linear), epsilonDistance)
	return clampFloat(angular, -maxAngular, maxAngular)
}
