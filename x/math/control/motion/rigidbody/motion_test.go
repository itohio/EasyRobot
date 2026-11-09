package rigidbody

import (
	"errors"
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	testStateRows   = stateVectorSize
	testControlRows = controlBaseSize
)

func TestNewMotionValidation(t *testing.T) {
	_, err := NewMotion(Constraints{}, Parameters{})
	if err == nil {
		t.Fatalf("expected error for invalid config")
	}
}

func TestForwardStraightPath(t *testing.T) {
	m := mustPlanner(t)
	path := []vec.Vector3D{{0, 0, 0}, {5, 0, 0}, {10, 0, 0}}
	if err := m.SetPath(path); err != nil {
		t.Fatalf("SetPath error: %v", err)
	}

	state := mat.New(testStateRows, 1)
	controls := mat.New(testControlRows, 1)
	destination := mat.New(testStateRows, 1)

	if err := m.Forward(state, destination, controls); err != nil {
		t.Fatalf("Forward error: %v", err)
	}

	if destination[0][0] <= state[0][0] {
		t.Errorf("expected forward progress along X, got %f", destination[0][0])
	}
	if math32.Abs(destination[3][0]) > 1e-3 {
		t.Errorf("expected yaw near zero, got %f", destination[3][0])
	}
	if destination[4][0] <= 0 || destination[4][0] > m.constraints.MaxSpeed {
		t.Errorf("speed out of bounds: %f", destination[4][0])
	}
	if math32.Abs(controls[1][0]) > m.constraints.MaxTurnRate {
		t.Errorf("angular control exceeded limits: %f", controls[1][0])
	}
	expectedTs := state[5][0] + m.params.SamplePeriod
	if math32.Abs(destination[5][0]-expectedTs) > 1e-6 {
		t.Errorf("timestamp mismatch: got %f want %f", destination[5][0], expectedTs)
	}
}

func TestForwardRequiresPath(t *testing.T) {
	m := mustPlanner(t)
	state := mat.New(testStateRows, 1)
	destination := mat.New(testStateRows, 1)
	controls := mat.New(testControlRows, 1)

	err := m.Forward(state, destination, controls)
	if !errors.Is(err, ErrInvalidPath) {
		t.Fatalf("expected ErrInvalidPath, got %v", err)
	}
}

func TestBackwardControlClamping(t *testing.T) {
	m := mustPlanner(t)

	state := mat.New(testStateRows, 1)
	destination := mat.New(testStateRows, 1)
	destination[0][0] = 1
	destination[4][0] = m.constraints.MaxSpeed * 2

	controls := mat.New(testControlRows, 1)

	if err := m.Backward(state, destination, controls); err != nil {
		t.Fatalf("Backward error: %v", err)
	}
	if controls[0][0] > m.constraints.MaxSpeed {
		t.Errorf("linear exceeded max speed: %f", controls[0][0])
	}
	if math32.Abs(controls[1][0]) > m.constraints.MaxTurnRate {
		t.Errorf("angular exceeded max turn rate: %f", controls[1][0])
	}
}

func TestMotionMetadata(t *testing.T) {
	m := mustPlanner(t)

	dims := m.Dimensions()
	if dims.StateRows != testStateRows || dims.ControlSize != testControlRows {
		t.Fatalf("unexpected dimensions: %+v", dims)
	}

	caps := m.Capabilities()
	if caps.Holonomic {
		t.Errorf("expected non-holonomic capability")
	}
	if caps.ConstraintRank != testControlRows {
		t.Errorf("unexpected constraint rank: %d", caps.ConstraintRank)
	}

	cs := m.ConstraintSet()
	if cs.ControlLower == nil || cs.ControlUpper == nil {
		t.Fatalf("expected constraint matrices to be populated")
	}
	if cs.ControlLower.Rows() != testControlRows {
		t.Fatalf("unexpected control lower rows: %d", cs.ControlLower.Rows())
	}
}

func TestSetWaypointMatrixPosition(t *testing.T) {
	m := mustPlanner(t)
	path := mat.New(3, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[0][1], path[1][1], path[2][1] = 5, 0, 0
	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix position failed: %v", err)
	}
}

func TestSetWaypointMatrixPose(t *testing.T) {
	m := customPlanner(t, Constraints{
		MaxSpeed:               5,
		MaxAcceleration:        5,
		MaxDeceleration:        5,
		MaxJerk:                10,
		MaxTurnRate:            math32.Pi,
		MaxTurnAcceleration:    math32.Pi * 4,
		MaxLateralAcceleration: 10,
	}, defaultParams())
	path := mat.New(7, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[3][0], path[4][0], path[5][0], path[6][0] = 1, 0, 0, 0
	path[0][1], path[1][1], path[2][1] = 0, 5, 0
	yaw := math32.Pi / 2
	path[3][1] = math32.Cos(yaw / 2)
	path[4][1] = 0
	path[5][1] = 0
	path[6][1] = math32.Sin(yaw / 2)

	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix pose failed: %v", err)
	}

	planner, ok := m.planner.(*posePlanner)
	if !ok {
		t.Fatalf("expected posePlanner, got %T", m.planner)
	}
	sample := planner.Sample(planner.Length())
	if math32.Abs(yawFromQuaternion(sample.orientation)-yaw) > 1e-4 {
		t.Fatalf("expected terminal yaw %.2f, got %.2f", yaw, yawFromQuaternion(sample.orientation))
	}
}

func TestSetWaypointMatrixPositionVelocity(t *testing.T) {
	m := customPlanner(t, Constraints{
		MaxSpeed:               10,
		MaxAcceleration:        10,
		MaxDeceleration:        10,
		MaxJerk:                20,
		MaxTurnRate:            math32.Pi,
		MaxTurnAcceleration:    math32.Pi * 2,
		MaxLateralAcceleration: 10,
	}, defaultParams())
	path := mat.New(6, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[3][0], path[4][0], path[5][0] = 0, 0, 0
	path[0][1], path[1][1], path[2][1] = 5, 0, 0
	path[3][1], path[4][1], path[5][1] = 4, 0, 0

	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix position velocity failed: %v", err)
	}

	planner, ok := m.planner.(*positionVelocityPlanner)
	if !ok {
		t.Fatalf("expected positionVelocityPlanner, got %T", m.planner)
	}
	sample := planner.Sample(planner.Length())
	if sample.linearHint[0] != 4 {
		t.Fatalf("expected terminal linear hint 4, got %.2f", sample.linearHint[0])
	}
}

func TestSetWaypointMatrixPoseVelocity(t *testing.T) {
	m := customPlanner(t, Constraints{
		MaxSpeed:               10,
		MaxAcceleration:        10,
		MaxDeceleration:        10,
		MaxJerk:                20,
		MaxTurnRate:            math32.Pi,
		MaxTurnAcceleration:    math32.Pi * 4,
		MaxLateralAcceleration: 10,
	}, defaultParams())
	path := mat.New(14, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[3][0], path[4][0], path[5][0], path[6][0] = 1, 0, 0, 0
	path[7][0], path[8][0], path[9][0] = 0, 0, 0
	path[10][0], path[11][0], path[12][0] = 0, 0, 0
	path[13][0] = 0

	path[0][1], path[1][1], path[2][1] = 0, 5, 0
	yaw := math32.Pi / 2
	path[3][1] = math32.Cos(yaw / 2)
	path[4][1] = 0
	path[5][1] = 0
	path[6][1] = math32.Sin(yaw / 2)
	path[7][1], path[8][1], path[9][1] = 3, 0, 0
	path[10][1], path[11][1], path[12][1] = 0, 0, 0.5
	path[13][1] = 0

	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix pose velocity failed: %v", err)
	}

	planner, ok := m.planner.(*poseVelocityPlanner)
	if !ok {
		t.Fatalf("expected poseVelocityPlanner, got %T", m.planner)
	}
	sample := planner.Sample(planner.Length())
	if math32.Abs(yawFromQuaternion(sample.orientation)-yaw) > 1e-4 {
		t.Fatalf("expected terminal yaw %.2f, got %.2f", yaw, yawFromQuaternion(sample.orientation))
	}
	if sample.linearHint[0] != 3 {
		t.Fatalf("expected terminal linear hint 3, got %.2f", sample.linearHint[0])
	}
	if sample.angularHint[2] != 0.5 {
		t.Fatalf("expected terminal angular hint 0.5, got %.2f", sample.angularHint[2])
	}
}

func mustPlanner(t *testing.T) *Motion {
	t.Helper()
	constraints := Constraints{
		MaxSpeed:               2,
		MaxAcceleration:        1,
		MaxDeceleration:        1,
		MaxJerk:                5,
		MaxTurnRate:            math32.Pi / 2,
		MaxTurnAcceleration:    math32.Pi,
		MaxLateralAcceleration: 3,
	}
	params := Parameters{
		SamplePeriod:      0.02,
		LookaheadDistance: 0.5,
		SpeedPID:          PIDGains{P: 1.0, I: 0.0, D: 0.1},
		HeadingPID:        PIDGains{P: 1.0, I: 0.0, D: 0.1},
		LateralPID:        PIDGains{P: 0.5, I: 0.0, D: 0.1},
	}

	m, err := NewMotion(constraints, params)
	if err != nil {
		t.Fatalf("failed to create motion planner: %v", err)
	}
	return m
}

func customPlanner(t *testing.T, constraints Constraints, params Parameters) *Motion {
	m, err := NewMotion(constraints, params)
	if err != nil {
		t.Fatalf("failed to create motion planner: %v", err)
	}
	return m
}

func defaultParams() Parameters {
	return Parameters{
		SamplePeriod:      0.02,
		LookaheadDistance: 0.5,
		SpeedPID:          PIDGains{P: 1.0, I: 0.0, D: 0.1},
		HeadingPID:        PIDGains{P: 1.0, I: 0.0, D: 0.1},
		LateralPID:        PIDGains{P: 0.5, I: 0.0, D: 0.1},
	}
}

// forwardTick is one closed-loop Forward step: the commanded position, the
// reported speed and the profile progress after the step.
type forwardTick struct {
	pos             vec.Vector3D
	speed, progress float32
}

// forwardRun drives Forward closed-loop from the origin (each destination
// state is fed back as the next state) for steps ticks.
func forwardRun(t *testing.T, m *Motion, steps int) []forwardTick {
	t.Helper()
	state := mat.New(testStateRows, 1)
	dst := mat.New(testStateRows, 1)
	controls := mat.New(testControlRows, 1)
	ticks := make([]forwardTick, 0, steps)
	for range steps {
		if err := m.Forward(state, dst, controls); err != nil {
			t.Fatalf("Forward: %v", err)
		}
		ticks = append(ticks, forwardTick{
			pos:      vec.Vector3D{dst[0][0], dst[1][0], dst[2][0]},
			speed:    dst[4][0],
			progress: m.progress,
		})
		for r := range testStateRows {
			state[r][0] = dst[r][0]
		}
	}
	return ticks
}

// displacementSpeeds returns the per-tick displacement speed |Δpos|/dt of a
// run starting at the origin.
func displacementSpeeds(ticks []forwardTick, dt float32) []float32 {
	speeds := make([]float32, len(ticks))
	var prev vec.Vector3D
	for i, tk := range ticks {
		speeds[i] = distance(prev, tk.pos) / dt
		prev = tk.pos
	}
	return speeds
}

// TestForwardStraightPathReachesCruiseSpeed drives Forward closed-loop along a
// straight 20 m path. The commanded positions must move at the configured
// MaxSpeed (per-tick displacement), the reported speed must match that motion,
// and the path must be covered within 15% of the time-optimal jerk-limited
// rest-to-rest duration, independent of the lookahead distance.
func TestForwardStraightPathReachesCruiseSpeed(t *testing.T) {
	const length = 20
	constraints := []Constraints{
		{MaxSpeed: 2, MaxAcceleration: 1, MaxDeceleration: 1, MaxJerk: 5, MaxTurnRate: math32.Pi / 2, MaxTurnAcceleration: math32.Pi, MaxLateralAcceleration: 3},
		{MaxSpeed: 10, MaxAcceleration: 10, MaxDeceleration: 10, MaxJerk: 20, MaxTurnRate: math32.Pi, MaxTurnAcceleration: 2 * math32.Pi, MaxLateralAcceleration: 10},
	}
	for _, c := range constraints {
		// Both profiles reach MaxSpeed and full acceleration (V*J >= A^2):
		// ramp time V/A + A/J covering V*ramp/2 each way, cruise in between.
		ramp := c.MaxSpeed/c.MaxAcceleration + c.MaxAcceleration/c.MaxJerk
		optimal := 2*ramp + (length-c.MaxSpeed*ramp)/c.MaxSpeed
		for _, lookahead := range []float32{0.5, 2} {
			params := Parameters{
				SamplePeriod:      0.02,
				LookaheadDistance: lookahead,
				SpeedPID:          PIDGains{P: 1, D: 0.1},
				HeadingPID:        PIDGains{P: 1, D: 0.1},
				LateralPID:        PIDGains{P: 0.5, D: 0.1},
			}
			m, err := NewMotion(c, params)
			if err != nil {
				t.Fatalf("NewMotion: %v", err)
			}
			if err := m.SetPath([]vec.Vector3D{{0, 0, 0}, {10, 0, 0}, {length, 0, 0}}); err != nil {
				t.Fatalf("SetPath: %v", err)
			}
			ticks := forwardRun(t, m, 5000)
			speeds := displacementSpeeds(ticks, params.SamplePeriod)
			var peak, peakReported, mismatch float32
			arrived := float32(-1)
			for i, tk := range ticks {
				peak = math32.Max(peak, speeds[i])
				peakReported = math32.Max(peakReported, tk.speed)
				// The report is the speed at the end of the tick, the
				// displacement its average: they differ by up to a*dt/2.
				mismatch = math32.Max(mismatch, math32.Abs(tk.speed-speeds[i]))
				if arrived < 0 && tk.pos[0] >= length-0.01 {
					arrived = float32(i+1) * params.SamplePeriod
				}
			}
			name := fmt.Sprintf("MaxSpeed %v lookahead %v", c.MaxSpeed, lookahead)
			t.Logf("%s: peak displacement speed %v (reported %v), arrived %vs, time-optimal %vs", name, peak, peakReported, arrived, optimal)
			if peak < 0.95*c.MaxSpeed || peak > c.MaxSpeed*1.001 {
				t.Errorf("%s: peak displacement speed %v, want within [0.95, 1.001] x MaxSpeed", name, peak)
			}
			if tol := c.MaxAcceleration * params.SamplePeriod; mismatch > tol {
				t.Errorf("%s: reported speed differs from the displacement speed by %v > %v", name, mismatch, tol)
			}
			if arrived < 0 || arrived > 1.15*optimal {
				t.Errorf("%s: arrived at %vs, want <= 1.15 x time-optimal %vs", name, arrived, optimal)
			}
			if final := ticks[len(ticks)-1].pos[0]; math32.Abs(final-length) > 1e-3 {
				t.Errorf("%s: final x %v, want %v", name, final, length)
			}
		}
	}
}

// TestForwardPathEndRespectsMaxDeceleration uses a MaxDeceleration well below
// MaxAcceleration: the commanded motion must brake for the path end within
// MaxDeceleration (and accelerate within MaxAcceleration), and the reported
// speed, which limitSpeed clamps to the MaxDeceleration braking curve, must
// match the displacement speed.
func TestForwardPathEndRespectsMaxDeceleration(t *testing.T) {
	const length, window = 20, 5 // window: finite-difference span in ticks
	c := Constraints{MaxSpeed: 2, MaxAcceleration: 2, MaxDeceleration: 0.5, MaxJerk: 5, MaxTurnRate: math32.Pi / 2, MaxTurnAcceleration: math32.Pi, MaxLateralAcceleration: 3}
	params := defaultParams()
	dt := params.SamplePeriod
	m := customPlanner(t, c, params)
	if err := m.SetPath([]vec.Vector3D{{0, 0, 0}, {10, 0, 0}, {length, 0, 0}}); err != nil {
		t.Fatalf("SetPath: %v", err)
	}
	ticks := forwardRun(t, m, 3000)
	speeds := displacementSpeeds(ticks, dt)
	var maxAcc, maxDec, mismatch float32
	span := float32(window) * dt
	for k := 2 * window; k < len(ticks); k++ {
		now := (ticks[k].pos[0] - ticks[k-window].pos[0]) / span
		before := (ticks[k-window].pos[0] - ticks[k-2*window].pos[0]) / span
		acc := (now - before) / span
		maxAcc, maxDec = math32.Max(maxAcc, acc), math32.Max(maxDec, -acc)
	}
	for i, tk := range ticks {
		mismatch = math32.Max(mismatch, math32.Abs(tk.speed-speeds[i]))
	}
	t.Logf("max acceleration %v, max deceleration %v, reported/displacement speed mismatch %v", maxAcc, maxDec, mismatch)
	if maxDec > 1.05*c.MaxDeceleration {
		t.Errorf("deceleration %v > 1.05 x MaxDeceleration %v", maxDec, c.MaxDeceleration)
	}
	if maxAcc > 1.05*c.MaxAcceleration {
		t.Errorf("acceleration %v > 1.05 x MaxAcceleration %v", maxAcc, c.MaxAcceleration)
	}
	if tol := c.MaxAcceleration * dt; mismatch > tol {
		t.Errorf("reported speed differs from the displacement speed by %v > %v", mismatch, tol)
	}
	if final := ticks[len(ticks)-1].pos[0]; math32.Abs(final-length) > 1e-3 {
		t.Errorf("final x %v, want %v", final, length)
	}
}

// TestForwardArcCapsSpeedAtLookahead follows a 10 m straight lead-in, a
// half-circle of radius 2 m (5° chords) and a straight lead-out. Wherever the
// lookahead point lies on the arc, the reported speed must respect the
// lateral-acceleration cap sqrt(MaxLateralAcceleration*R), including while
// the vehicle itself is still on the lead-in (the cap anticipates the
// curvature at the lookahead point). The three-point curvature estimate on
// the chords reads up to ~11% low between vertices, so the cap gets 10%. The
// commanded per-tick displacement never exceeds MaxSpeed.
func TestForwardArcCapsSpeedAtLookahead(t *testing.T) {
	const radius, lead, lookahead, chords = 2, 10, 2, 36
	c := Constraints{MaxSpeed: 2, MaxAcceleration: 1, MaxDeceleration: 1, MaxJerk: 5, MaxTurnRate: math32.Pi, MaxTurnAcceleration: 4 * math32.Pi, MaxLateralAcceleration: 0.5}
	params := defaultParams()
	params.LookaheadDistance = lookahead
	path := []vec.Vector3D{{0, 0, 0}}
	for i := range chords + 1 {
		phi := -math32.Pi/2 + math32.Pi*float32(i)/chords
		path = append(path, vec.Vector3D{lead + radius*math32.Cos(phi), radius + radius*math32.Sin(phi), 0})
	}
	path = append(path, vec.Vector3D{0, 2 * radius, 0})
	m := customPlanner(t, c, params)
	if err := m.SetPath(path); err != nil {
		t.Fatalf("SetPath: %v", err)
	}
	arcEnd := lead + chords*2*radius*math32.Sin(math32.Pi/(2*chords))
	// Curvature samples points this far either side of the lookahead point.
	sampling := math32.Max(0.01*m.planner.Length(), 0.05)
	capSpeed := math32.Sqrt(c.MaxLateralAcceleration * radius)

	ticks := forwardRun(t, m, 3000)
	speeds := displacementSpeeds(ticks, params.SamplePeriod)
	var onArc, beforeArc, violations int
	for i, tk := range ticks {
		if speeds[i] > 1.001*c.MaxSpeed {
			t.Errorf("tick %d: displacement speed %v > MaxSpeed", i, speeds[i])
		}
		ahead := tk.progress + lookahead
		if ahead < lead+sampling || ahead > arcEnd-sampling {
			continue
		}
		onArc++
		if tk.progress < lead {
			beforeArc++
		}
		if tk.speed > 1.1*capSpeed {
			if violations++; violations <= 5 {
				t.Errorf("tick %d: progress %v (lookahead on the arc): reported speed %v > 1.1 x lateral cap %v", i, tk.progress, tk.speed, capSpeed)
			}
		}
	}
	t.Logf("%d ticks with the lookahead on the arc, %d of them before the arc; %d over the cap", onArc, beforeArc, violations)
	if beforeArc == 0 {
		t.Fatalf("no tick had the lookahead on the arc while still on the lead-in")
	}
	if final := ticks[len(ticks)-1].pos; distance(final, path[len(path)-1]) > 1e-3 {
		t.Errorf("final position %v, want %v", final, path[len(path)-1])
	}
}
