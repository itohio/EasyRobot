package rigidbody

import (
	"fmt"
	"math"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	vaj "github.com/itohio/EasyRobot/x/math/control/motion"
	"github.com/itohio/EasyRobot/x/math/control/pid"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	epsilonDistance = 1e-4
	pidLimit        = float32(math.MaxFloat32)
)

var _ kintypes.Bidirectional = (*Motion)(nil)

// Motion maintains stateful trajectory planning utilities.
type Motion struct {
	constraints Constraints
	params      Parameters

	dimensions    kintypes.Dimensions
	capabilities  kintypes.Capabilities
	constraintSet kintypes.Constraints
	controlRows   int

	speedProfile vaj.VAJ1D
	speedPID     pid.PID1D
	headingPID   pid.PID1D
	lateralPID   pid.PID1D

	progress      float32
	pathSignature uint64
	planner       pathPlanner
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
		dimensions: kintypes.Dimensions{
			StateRows:    stateVectorSize,
			StateCols:    1,
			ControlSize:  controlBaseSize,
			ActuatorSize: controlBaseSize,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:        false,
			Omnidirectional:  false,
			Underactuated:    false,
			SupportsLateral:  false,
			SupportsVertical: false,
			ConstraintRank:   controlBaseSize,
		},
		constraintSet: buildConstraintSet(constraints),
		controlRows:   controlBaseSize,
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

func buildConstraintSet(c Constraints) kintypes.Constraints {
	lower := mat.New(controlBaseSize, 1)
	upper := mat.New(controlBaseSize, 1)
	rate := mat.New(controlBaseSize, 1)

	lower[controlIdxLinear][0] = -c.MaxSpeed
	upper[controlIdxLinear][0] = c.MaxSpeed
	lower[controlIdxAngular][0] = -c.MaxTurnRate
	upper[controlIdxAngular][0] = c.MaxTurnRate

	if c.MaxAcceleration > 0 {
		rate[controlIdxLinear][0] = c.MaxAcceleration
	}
	if c.MaxTurnAcceleration > 0 {
		rate[controlIdxAngular][0] = c.MaxTurnAcceleration
	}

	return kintypes.Constraints{
		ControlLower: lower,
		ControlUpper: upper,
		ControlRate:  rate,
	}
}

func (m *Motion) Dimensions() kintypes.Dimensions {
	return m.dimensions
}

func (m *Motion) Capabilities() kintypes.Capabilities {
	return m.capabilities
}

func (m *Motion) ConstraintSet() kintypes.Constraints {
	return m.constraintSet
}

func (m *Motion) SetPath(path []vec.Vector3D) error {
	planner, err := plannerFromPositions(path)
	if err != nil {
		return err
	}
	m.planner = planner
	m.handlePathChange(planner.Signature())
	return nil
}

// SetWaypointMatrix accepts a waypoint matrix where columns represent waypoints. Supported row layouts:
//   - 3 rows: position (x, y, z)
//   - 6 rows: position + linear velocity hints (vx, vy, vz)
//   - 7 rows: position + orientation quaternion (qw, qx, qy, qz)
//   - 14 rows: position + orientation quaternion + linear velocity hints + angular velocity hints (optional extra metadata rows ignored)
func (m *Motion) SetWaypointMatrix(path mattype.Matrix) error {
	if path == nil {
		return ErrInvalidPath
	}
	planner, err := plannerFromWaypointMatrix(path)
	if err != nil {
		return err
	}
	m.planner = planner
	m.handlePathChange(planner.Signature())
	return nil
}

func (m *Motion) resizeControls(rows int) {
	if rows == m.controlRows {
		return
	}
	if rows < controlBaseSize {
		rows = controlBaseSize
	}
	m.controlRows = rows
	m.dimensions.ControlSize = rows
	m.dimensions.ActuatorSize = rows
	lower := mat.New(rows, 1)
	upper := mat.New(rows, 1)
	rate := mat.New(rows, 1)

	lower[controlIdxLinear][0] = -m.constraints.MaxSpeed
	upper[controlIdxLinear][0] = m.constraints.MaxSpeed
	lower[controlIdxAngular][0] = -m.constraints.MaxTurnRate
	upper[controlIdxAngular][0] = m.constraints.MaxTurnRate

	if m.constraints.MaxAcceleration > 0 {
		rate[controlIdxLinear][0] = m.constraints.MaxAcceleration
	}
	if m.constraints.MaxTurnAcceleration > 0 {
		rate[controlIdxAngular][0] = m.constraints.MaxTurnAcceleration
	}

	m.constraintSet.ControlLower = lower
	m.constraintSet.ControlUpper = upper
	m.constraintSet.ControlRate = rate
}

// Forward plans the next trajectory segment given current state and controls.
func (m *Motion) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if m.planner == nil {
		return ErrInvalidPath
	}
	if err := ensureStateMatrix(state); err != nil {
		return err
	}
	if err := ensureDestinationMatrix(destination); err != nil {
		return err
	}

	currentState, err := stateFromMatrix(state)
	if err != nil {
		return err
	}

	inputControls := m.lastControls
	if controls != nil {
		rows := controls.Rows()
		if rows < controlBaseSize {
			return fmt.Errorf("planner: %w (controls rows %d)", kintypes.ErrInvalidDimensions, rows)
		}
		m.resizeControls(rows)
		if err := ensureControlsMatrix(controls, m.controlRows); err != nil {
			return err
		}
		inputControls, err = controlsFromMatrix(controls, m.controlRows)
		if err != nil {
			return err
		}
	}

	nextState, nextControls, err := m.planForward(currentState, inputControls)
	if err != nil {
		return err
	}
	if err := populateStateMatrix(destination, nextState); err != nil {
		return err
	}
	if controls != nil {
		if err := populateControlsMatrix(controls, nextControls, m.controlRows); err != nil {
			return err
		}
	}
	return nil
}

func (m *Motion) planForward(state State, controls Controls) (State, Controls, error) {
	planner := m.planner
	if planner == nil {
		return State{}, Controls{}, ErrInvalidPath
	}
	total := planner.Length()
	if total <= 0 {
		return State{}, Controls{}, ErrInvalidPath
	}
	m.handlePathChange(planner.Signature())

	m.progress = m.projectProgress(state.Position)
	target := clampFloat(m.progress+m.params.LookaheadDistance, 0, total)

	profile := m.advanceProfile(target, total)
	sample := planner.Sample(profile.progress)
	curvature := planner.Curvature(profile.progress)
	remaining := total - profile.progress
	speed := m.limitSpeed(profile.velocity, curvature, remaining)

	targetYaw := normalizeAngle(math32.Atan2(sample.tangent[1], sample.tangent[0]))
	if sample.hasOrientation {
		targetYaw = normalizeAngle(yawFromQuaternion(sample.orientation))
	}
	nextYaw, yawRate := m.advanceYaw(state.Yaw, targetYaw)

	if sample.hasAngularHint {
		yawHint := clampFloat(sample.angularHint[2], -m.constraints.MaxTurnRate, m.constraints.MaxTurnRate)
		yawHint = m.applyAngularAcceleration(yawHint, m.params.SamplePeriod, m.lastYawRate)
		m.lastYawRate = yawHint
		yawRate = yawHint
		nextYaw = normalizeAngle(state.Yaw + yawRate*m.params.SamplePeriod)
	}

	if sample.hasLinearHint {
		tangentMag := vectorMagnitude(sample.tangent)
		if tangentMag > epsilonDistance {
			projected := dot(sample.linearHint, sample.tangent) / tangentMag
			if projected > epsilonDistance {
				speed = clampFloat(projected, 0, m.constraints.MaxSpeed)
			}
		} else {
			hintMag := vectorMagnitude(sample.linearHint)
			if hintMag > epsilonDistance {
				speed = clampFloat(hintMag, 0, m.constraints.MaxSpeed)
			}
		}
	}

	next := State{
		Position:  sample.position,
		Yaw:       nextYaw,
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

	return next, nextControls, nil
}

// Backward computes control commands to follow the desired trajectory.
func (m *Motion) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if destination == nil {
		return ErrInvalidPath
	}
	if controls == nil {
		return fmt.Errorf("planner: %w (controls required)", kintypes.ErrInvalidDimensions)
	}
	rows := controls.Rows()
	if rows < controlBaseSize {
		return fmt.Errorf("planner: %w (controls rows %d)", kintypes.ErrInvalidDimensions, rows)
	}
	m.resizeControls(rows)
	if err := ensureControlsMatrix(controls, m.controlRows); err != nil {
		return err
	}

	desiredState, err := stateFromMatrix(destination)
	if err != nil {
		return err
	}

	var currentState State
	if state != nil {
		currentState, err = stateFromMatrix(state)
		if err != nil {
			return err
		}
	} else {
		currentState = m.lastState
	}

	inputControls, err := controlsFromMatrix(controls, m.controlRows)
	if err != nil {
		return err
	}

	outputControls, err := m.planBackward(Trajectory{States: []State{desiredState}}, currentState, inputControls)
	if err != nil {
		return err
	}

	return populateControlsMatrix(controls, outputControls, m.controlRows)
}

func (m *Motion) planBackward(tr Trajectory, state State, controls Controls) (Controls, error) {
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

func (m *Motion) projectProgress(position vec.Vector3D) float32 {
	if m.planner == nil {
		return 0
	}
	return m.planner.Project(position)
}

func (m *Motion) advanceProfile(target, total float32) profileSnapshot {
	if total <= 0 {
		return profileSnapshot{}
	}
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

func (m *Motion) advanceYaw(currentYaw, targetYaw float32) (float32, float32) {
	dt := m.params.SamplePeriod
	maxDelta := m.constraints.MaxTurnRate * dt
	delta := clampFloat(normalizeAngle(targetYaw-currentYaw), -maxDelta, maxDelta)
	yawRate := delta / dt
	if m.constraints.MaxTurnAcceleration > 0 {
		yawRate = m.applyAngularAcceleration(yawRate, dt, m.lastYawRate)
	}
	m.lastYawRate = yawRate
	nextYaw := normalizeAngle(currentYaw + yawRate*dt)
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
