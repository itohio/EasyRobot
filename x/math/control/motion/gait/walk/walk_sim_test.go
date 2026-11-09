package walk

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/control/kinematics/chain"
	"github.com/itohio/EasyRobot/x/math/control/motion/gait/scheduler"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// M-a stage 1: host simulation of a hexapod tripod with the probe-R leg.
// 10 s of 0.1 m/s with ω = 0.3 rad/s, then 5 s turning in place at 0.5 rad/s,
// 20 ms ticks. Every tick (checked in run, from the servo commands): stance
// feet fixed in the world (< 1 mm), every target solved without an IK error,
// every joint inside its limits, every servo command inside its travel, and
// the commanded foot on its target. Reports the maximum joint speed.
func TestHexapodTripodSim(t *testing.T) {
	cfg := hexConfig()
	w := newWalker(t, cfg)
	st := &simStats{}
	walk := Twist{VX: 0.1, W: 0.3}
	turn := Twist{W: 0.5}
	run(t, w, 10, func(float32) Twist { return walk }, st)
	walkPose, walkScale := w.Pose(), w.Scale()
	run(t, w, 5, func(float32) Twist { return turn }, st)

	maxVel := cfg.Body.Mounts[0].Leg.Joints[0].MaxVel
	t.Logf("ticks %d; after walk pose %+v (scale %.3f), final pose %+v (scale %.3f)", st.ticks, walkPose, walkScale, w.Pose(), w.Scale())
	t.Logf("max stance drift %.3e m; min reach margin %.4f m (clip margin %.3f); max joint speed coxa %.2f (leg %d) femur %.2f (leg %d) tibia %.2f (leg %d) rad/s, limit %.2f rad/s; apex %.4f m",
		st.maxDrift, st.minMargin, cfg.Margin, st.maxSpeed[0], st.maxSpeedLeg[0], st.maxSpeed[1], st.maxSpeedLeg[1], st.maxSpeed[2], st.maxSpeedLeg[2], maxVel, st.maxApex)

	assert.Equal(t, 750, st.ticks)
	assert.Less(t, st.maxDrift, float32(1e-3), "stance feet fixed in the world")
	assert.Zero(t, st.ikErrors, "IK errors")
	assert.Zero(t, st.limitErrors, "joints outside limits")
	assert.Zero(t, st.servoErrors, "servo commands outside travel")
	assert.Greater(t, st.minMargin, float32(0), "every foot target reachable")
	assert.Equal(t, 3, st.minStanceCnt)
	assert.Equal(t, 3, st.maxStanceCnt)
	assert.Equal(t, float32(1), walkScale, "0.1 m/s + 0.3 rad/s fits the reach without clipping")
	// The body follows an arc of radius v/ω = 1/3 m: ω·t of yaw (the
	// 0 → 0.3 rad/s ramp at 1.5 rad/s² costs 0.1 s), ending ≈ 2R = 0.66 m left.
	assert.InDelta(t, 0.3*(10-0.1), walkPose.Yaw, 0.02)
	assert.InDelta(t, 0.66, walkPose.Y, 0.02)

	servo := make([]float32, 18)
	allocs := testing.AllocsPerRun(100, func() { _ = w.Step(tick, turn, servo) })
	t.Logf("allocations per tick: %v", allocs)
	assert.Zero(t, allocs)
}

// A quadruped running the walk preset (β = 3/4): kinematics only, no
// stability assertion (that is ER-037).
func TestQuadrupedWalkSim(t *testing.T) {
	cfg := Config{
		Body:         chain.Quadruped(0.12, 0.06, probeR),
		Pattern:      scheduler.Walk(),
		Period:       1.6,
		SwingHeight:  0.03,
		LiftFraction: 0.1,
		Margin:       0.008,
		MaxAccel:     0.3,
		MaxAngAccel:  1,
	}
	w, err := New(cfg)
	require.NoError(t, err)
	st := &simStats{}
	run(t, w, 8, func(float32) Twist { return Twist{VX: 0.05, W: 0.1} }, st)
	run(t, w, 4, func(float32) Twist { return Twist{W: 0.3} }, st)
	t.Logf("quadruped: final pose %+v scale %.3f; max stance drift %.3e m; min margin %.4f m; max joint speed %.2f/%.2f/%.2f rad/s; stance legs %d…%d",
		w.Pose(), w.Scale(), st.maxDrift, st.minMargin, st.maxSpeed[0], st.maxSpeed[1], st.maxSpeed[2], st.minStanceCnt, st.maxStanceCnt)
	assert.Less(t, st.maxDrift, float32(1e-3))
	assert.Zero(t, st.ikErrors)
	assert.Zero(t, st.limitErrors)
	assert.Zero(t, st.servoErrors)
	assert.Equal(t, 3, st.minStanceCnt)
	assert.Equal(t, 3, st.maxStanceCnt, "walk: one leg in swing at a time")
	assert.Greater(t, w.Pose().X, float32(0.2))
}
