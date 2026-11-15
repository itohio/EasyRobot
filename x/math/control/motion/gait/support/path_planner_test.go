package support

import (
	"context"
	"testing"
	"time"

	gaittypes "github.com/itohio/EasyRobot/x/math/control/motion/gait/types"
	vec "github.com/itohio/EasyRobot/x/math/vec"
)

// swingTrace holds foot positions sampled over one stance→lift→swing→touchdown cycle.
type swingTrace struct {
	lift, swing, touchdown []vec.Vector3D
	stanceEnd              vec.Vector3D
}

// runSwingCycle drives one leg through support (to stance), lift, swing and touchdown
// with `steps` samples per phase, the way a phase scheduler would.
func runSwingCycle(t *testing.T, profile gaittypes.SupportProfile, stance, foothold vec.Vector3D, steps int) swingTrace {
	t.Helper()
	planner := NewSupportPathPlanner(WithSupportProfile(profile))
	ctx := context.Background()
	now := time.Unix(0, 0)
	const dt = 10 * time.Millisecond

	run := func(mode gaittypes.PhaseMode, desired vec.Vector3D) []vec.Vector3D {
		out := make([]vec.Vector3D, 0, steps+1)
		for i := range steps + 1 {
			now = now.Add(dt)
			fs, err := planner.Update(ctx, gaittypes.SupportUpdateRequest{
				LegID:       "L1",
				DesiredPose: gaittypes.EndpointPose{Position: desired},
				Phase:       gaittypes.PhaseState{Mode: mode, Progress: float32(i) / float32(steps)},
				Timestamp:   now,
				Delta:       dt,
			})
			if err != nil {
				t.Fatalf("Update(%d, %d/%d): %v", mode, i, steps, err)
			}
			out = append(out, fs.Pose.Position)
		}
		return out
	}

	support := run(gaittypes.PhaseSupport, stance)
	return swingTrace{
		stanceEnd: support[len(support)-1],
		lift:      run(gaittypes.PhaseLift, foothold),
		swing:     run(gaittypes.PhaseSwing, foothold),
		touchdown: run(gaittypes.PhaseTouchdown, foothold),
	}
}

func maxZ(trace ...[]vec.Vector3D) float32 {
	m := float32(-1e9)
	for _, seg := range trace {
		for _, p := range seg {
			if p[2] > m {
				m = p[2]
			}
		}
	}
	return m
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

var swingProfiles = []struct {
	name    string
	profile gaittypes.SupportProfile
	stance  vec.Vector3D
}{
	{"default ground 0", gaittypes.SupportProfile{LiftHeight: 0.05, SwingHeight: 0.08, SupportBlend: 1}, vec.Vector3D{0, 0, 0}},
	{"hexapod body frame", gaittypes.SupportProfile{LiftHeight: 0.01, SwingHeight: 0.03, SupportBlend: 1}, vec.Vector3D{0.12, 0.05, -0.08}},
	{"no vertical lift segment", gaittypes.SupportProfile{LiftHeight: 0, SwingHeight: 0.04, SupportBlend: 1}, vec.Vector3D{0.1, -0.02, -0.06}},
	{"lift equals swing", gaittypes.SupportProfile{LiftHeight: 0.03, SwingHeight: 0.03, SupportBlend: 1}, vec.Vector3D{0, 0, -0.07}},
	{"lift above swing is clamped", gaittypes.SupportProfile{LiftHeight: 0.05, SwingHeight: 0.03, SupportBlend: 1}, vec.Vector3D{0.1, 0.1, -0.09}},
	{"touchdown clearance without lift segment", gaittypes.SupportProfile{LiftHeight: 0, SwingHeight: 0.04, TouchdownClearance: 0.01, SupportBlend: 1}, vec.Vector3D{0.1, 0, -0.06}},
	{"touchdown clearance above lift", gaittypes.SupportProfile{LiftHeight: 0.01, SwingHeight: 0.04, TouchdownClearance: 0.02, SupportBlend: 1}, vec.Vector3D{0.1, 0, -0.06}},
}

// The swing apex is SwingHeight above the lift-off (stance) height, ±1 mm.
func TestSwingApexEqualsSwingHeight(t *testing.T) {
	for _, tc := range swingProfiles {
		t.Run(tc.name, func(t *testing.T) {
			foothold := tc.stance
			foothold[0] += 0.1 // stride 0.1 m
			tr := runSwingCycle(t, tc.profile, tc.stance, foothold, 100)
			apex := maxZ(tr.lift, tr.swing, tr.touchdown) - tr.stanceEnd[2]
			if abs32(apex-tc.profile.SwingHeight) > 1e-3 {
				t.Fatalf("apex %.4f m above stance, want SwingHeight %.4f ± 0.001 (lift peak %.4f, swing peak %.4f, touchdown peak %.4f)",
					apex, tc.profile.SwingHeight,
					maxZ(tr.lift)-tr.stanceEnd[2], maxZ(tr.swing)-tr.stanceEnd[2], maxZ(tr.touchdown)-tr.stanceEnd[2])
			}
		})
	}
}

// Touchdown z never rises; it descends monotonically onto the foothold
// (+ the normalized TouchdownClearance).
func TestTouchdownMonotonicDescent(t *testing.T) {
	for _, tc := range swingProfiles {
		t.Run(tc.name, func(t *testing.T) {
			foothold := tc.stance
			foothold[0] += 0.1
			tr := runSwingCycle(t, tc.profile, tc.stance, foothold, 100)
			for i := 1; i < len(tr.touchdown); i++ {
				if dz := tr.touchdown[i][2] - tr.touchdown[i-1][2]; dz > 1e-6 {
					t.Fatalf("touchdown z rises at sample %d: %.4f -> %.4f (peak %.4f)",
						i, tr.touchdown[i-1][2], tr.touchdown[i][2], maxZ(tr.touchdown))
				}
			}
			end := tr.touchdown[len(tr.touchdown)-1]
			want := foothold
			want[2] += tc.profile.Normalize().TouchdownClearance
			for k := range 3 {
				if abs32(end[k]-want[k]) > 1e-6 {
					t.Fatalf("touchdown ends at %v, want %v", end, want)
				}
			}
		})
	}
}

// Lift starts at the stance height (no jump) and rises monotonically to
// stance + LiftHeight without overshoot.
func TestLiftStartsAtStanceHeight(t *testing.T) {
	for _, tc := range swingProfiles {
		t.Run(tc.name, func(t *testing.T) {
			foothold := tc.stance
			foothold[0] += 0.1
			tr := runSwingCycle(t, tc.profile, tc.stance, foothold, 100)
			for k := range 3 {
				if abs32(tr.lift[0][k]-tr.stanceEnd[k]) > 1e-6 {
					t.Fatalf("lift starts at %v, stance ended at %v", tr.lift[0], tr.stanceEnd)
				}
			}
			top := tr.stanceEnd[2] + tc.profile.Normalize().LiftHeight
			for i, p := range tr.lift {
				if i > 0 && p[2] < tr.lift[i-1][2]-1e-6 {
					t.Fatalf("lift z falls at sample %d: %.4f -> %.4f", i, tr.lift[i-1][2], p[2])
				}
				if p[2] > top+1e-6 {
					t.Fatalf("lift z %.4f at sample %d exceeds stance+LiftHeight %.4f", p[2], i, top)
				}
			}
			if end := tr.lift[len(tr.lift)-1][2]; abs32(end-top) > 1e-6 {
				t.Fatalf("lift ends at z %.4f, want stance+LiftHeight %.4f", end, top)
			}
		})
	}
}

// Every phase starts where the previous one ended and the foot never teleports.
func TestSwingCycleContinuity(t *testing.T) {
	for _, tc := range swingProfiles {
		t.Run(tc.name, func(t *testing.T) {
			foothold := tc.stance
			foothold[0] += 0.1
			tr := runSwingCycle(t, tc.profile, tc.stance, foothold, 100)
			pairs := []struct {
				name     string
				end, beg vec.Vector3D
			}{
				{"stance->lift", tr.stanceEnd, tr.lift[0]},
				{"lift->swing", tr.lift[len(tr.lift)-1], tr.swing[0]},
				{"swing->touchdown", tr.swing[len(tr.swing)-1], tr.touchdown[0]},
			}
			for _, pr := range pairs {
				for k := range 3 {
					if abs32(pr.end[k]-pr.beg[k]) > 1e-6 {
						t.Fatalf("%s jumps: %v -> %v", pr.name, pr.end, pr.beg)
					}
				}
			}
			// Swing ends above the foothold, so touchdown is a vertical descent.
			swingEnd := tr.swing[len(tr.swing)-1]
			if abs32(swingEnd[0]-foothold[0]) > 1e-6 || abs32(swingEnd[1]-foothold[1]) > 1e-6 {
				t.Fatalf("swing ends at %v, want above foothold %v", swingEnd, foothold)
			}
		})
	}
}
