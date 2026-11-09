package scheduler

import (
	"math"
	"testing"
	"time"

	gaittypes "github.com/itohio/EasyRobot/x/math/control/motion/gait/types"
)

func mustNew(t *testing.T, p Pattern, period float32) *Scheduler {
	t.Helper()
	s, err := New(p, period)
	if err != nil {
		t.Fatalf("New(%s, %v): %v", p.Name, period, err)
	}
	return s
}

// boundaryPhases are the hand-off instants of every preset (multiples of 1/12 cover
// 1/2, 1/3, 1/4 and 1/6) plus their float32 neighbours.
func boundaryPhases() []float32 {
	var out []float32
	for k := range 12 {
		p := float32(k) / 12
		out = append(out, p, math.Nextafter32(p, 1))
		if k > 0 {
			out = append(out, math.Nextafter32(p, 0))
		}
	}
	return append(out, math.Nextafter32(1, 0))
}

// sweep calls check on a dense grid of reset phases, on every boundary phase and
// along a long Advance run with a step incommensurate with the period.
func sweep(t *testing.T, s *Scheduler, check func(st State) string) {
	t.Helper()
	visit := func(how string) {
		st := s.State()
		if msg := check(st); msg != "" {
			t.Fatalf("%s, phase %.9f: %s; legs %+v", how, st.Phase, msg, st.Leg[:st.Legs])
		}
	}
	for k := range 4096 {
		s.Reset(float32(k) / 4096)
		visit("grid")
	}
	for _, p := range boundaryPhases() {
		s.Reset(p)
		visit("boundary")
	}
	s.Reset(0)
	for range 20000 {
		s.Advance(0.0137)
		visit("advance")
	}
}

// liftOffOrder advances over `cycles` cycles and returns the legs in lift-off order.
func liftOffOrder(s *Scheduler, cycles int) []int {
	const steps = 997
	s.Reset(0.001)
	prev := s.State()
	var order []int
	dt := s.Period() / steps
	for range cycles * steps {
		s.Advance(dt)
		st := s.State()
		for i := range st.Legs {
			if prev.Leg[i].Stance && !st.Leg[i].Stance {
				order = append(order, i)
			}
		}
		prev = st
	}
	return order
}

// isCyclic reports whether got is `want` repeated, starting at any rotation.
func isCyclic(got, want []int) bool {
	if len(got) < len(want) {
		return false
	}
	start := -1
	for i, w := range want {
		if w == got[0] {
			start = i
		}
	}
	if start < 0 {
		return false
	}
	for i, g := range got {
		if g != want[(start+i)%len(want)] {
			return false
		}
	}
	return true
}

func swingCount(st State, legs ...int) int {
	n := 0
	for _, i := range legs {
		if !st.Leg[i].Stance {
			n++
		}
	}
	return n
}

// T-IK-09: tripod — offsets {0, ½}, β = 0.5, always exactly three stance legs, and
// the groups {L1 R2 L3} / {R1 L2 R3} alternate.
func TestTripod(t *testing.T) {
	p := Tripod()
	groupA := []int{HexL1, HexR2, HexL3}
	groupB := []int{HexR1, HexL2, HexR3}
	if p.Legs != 6 {
		t.Fatalf("Tripod legs = %d, want 6", p.Legs)
	}
	for _, i := range groupA {
		if p.Offset[i] != 0 || p.Duty[i] != 0.5 {
			t.Fatalf("leg %d offset %v duty %v, want 0 / 0.5", i, p.Offset[i], p.Duty[i])
		}
	}
	for _, i := range groupB {
		if p.Offset[i] != 0.5 || p.Duty[i] != 0.5 {
			t.Fatalf("leg %d offset %v duty %v, want 0.5 / 0.5", i, p.Offset[i], p.Duty[i])
		}
	}
	s := mustNew(t, p, 1.1)
	sweep(t, s, func(st State) string {
		if n := st.StanceCount(); n != 3 {
			return "stance count != 3"
		}
		a := st.Leg[groupA[0]]
		for _, i := range groupA {
			if st.Leg[i] != a {
				return "group A legs differ"
			}
		}
		for _, i := range groupB {
			if st.Leg[i].Stance == a.Stance {
				return "group B in the same contact state as group A"
			}
		}
		return ""
	})
}

// T-IK-09: wave — β ≥ 5/6, at least five stance legs at all times, swings run back
// to front on the left side, then back to front on the right side.
func TestWave(t *testing.T) {
	p := Wave()
	for i := range p.Legs {
		if p.Duty[i] < 5.0/6 {
			t.Fatalf("leg %d duty %v < 5/6", i, p.Duty[i])
		}
	}
	s := mustNew(t, p, 1.2)
	sweep(t, s, func(st State) string {
		if st.StanceCount() < 5 {
			return "fewer than 5 stance legs"
		}
		return ""
	})
	want := []int{HexL3, HexL2, HexL1, HexR3, HexR2, HexR1}
	if got := liftOffOrder(s, 3); len(got) != 18 || !isCyclic(got, want) {
		t.Fatalf("lift-off order %v, want 3 cycles of %v", got, want)
	}
}

// Ripple — β = 2/3, at least four stance legs, at most one leg per side in swing,
// each side a back-to-front wave.
func TestRipple(t *testing.T) {
	p := Ripple()
	s := mustNew(t, p, 1.0)
	sweep(t, s, func(st State) string {
		if st.StanceCount() < 4 {
			return "fewer than 4 stance legs"
		}
		if swingCount(st, HexL1, HexL2, HexL3) > 1 || swingCount(st, HexR1, HexR2, HexR3) > 1 {
			return "two legs of one side in swing"
		}
		return ""
	})
	want := []int{HexL3, HexR1, HexL2, HexR3, HexL1, HexR2}
	if got := liftOffOrder(s, 3); len(got) != 18 || !isCyclic(got, want) {
		t.Fatalf("lift-off order %v, want 3 cycles of %v", got, want)
	}
}

// T-IK-09: trot — diagonal pairs LF+RH and RF+LH share phase, pairs alternate,
// exactly two stance legs.
func TestTrot(t *testing.T) {
	p := Trot()
	if p.Legs != 4 {
		t.Fatalf("Trot legs = %d, want 4", p.Legs)
	}
	s := mustNew(t, p, 0.6)
	sweep(t, s, func(st State) string {
		if st.Leg[QuadLF] != st.Leg[QuadRH] || st.Leg[QuadRF] != st.Leg[QuadLH] {
			return "diagonal pair out of phase"
		}
		if st.Leg[QuadLF].Stance == st.Leg[QuadRF].Stance {
			return "both diagonal pairs in the same contact state"
		}
		if st.StanceCount() != 2 {
			return "stance count != 2"
		}
		return ""
	})
}

// T-IK-09: walk — β = 0.75, exactly one leg in swing at a time, lateral sequence
// LH → LF → RH → RF.
func TestWalk(t *testing.T) {
	p := Walk()
	for i := range p.Legs {
		if p.Duty[i] != 0.75 {
			t.Fatalf("leg %d duty %v, want 0.75", i, p.Duty[i])
		}
	}
	s := mustNew(t, p, 1.6)
	sweep(t, s, func(st State) string {
		if st.Legs-st.StanceCount() != 1 {
			return "swing count != 1"
		}
		return ""
	})
	want := []int{QuadLH, QuadLF, QuadRH, QuadRF}
	if got := liftOffOrder(s, 3); len(got) != 12 || !isCyclic(got, want) {
		t.Fatalf("lift-off order %v, want 3 cycles of %v", got, want)
	}
}

// Phase advances by dt/period, wraps into [0,1), and every leg lifts off exactly
// once per cycle.
func TestPhaseWraps(t *testing.T) {
	const period, ticks, cycles = 0.8, 40, 10
	s := mustNew(t, Tripod(), period)
	s.Reset(0.1)
	prev := s.State()
	wraps := 0
	liftOffs := [MaxLegs]int{}
	for i := range ticks * cycles {
		s.Advance(period / ticks)
		st := s.State()
		if st.Phase < 0 || st.Phase >= 1 {
			t.Fatalf("tick %d: phase %v outside [0,1)", i, st.Phase)
		}
		if st.Phase < prev.Phase {
			wraps++
		}
		for leg := range st.Legs {
			if prev.Leg[leg].Stance && !st.Leg[leg].Stance {
				liftOffs[leg]++
			}
		}
		prev = st
	}
	if wraps != cycles {
		t.Fatalf("wraps = %d, want %d", wraps, cycles)
	}
	if d := s.Phase() - 0.1; d > 1e-5 || d < -1e-5 {
		t.Fatalf("after %d whole cycles phase = %v, want 0.1", cycles, s.Phase())
	}
	for leg := range 6 {
		if liftOffs[leg] != cycles {
			t.Fatalf("leg %d lifted off %d times in %d cycles", leg, liftOffs[leg], cycles)
		}
	}
	// Non-positive and non-finite steps are no-ops.
	before := s.Phase()
	for _, dt := range []float32{0, -0.1, float32(math.NaN()), float32(math.Inf(1))} {
		s.Advance(dt)
		if s.Phase() != before {
			t.Fatalf("Advance(%v) moved phase %v -> %v", dt, before, s.Phase())
		}
	}
}

// Stance progress runs 0 (touchdown) → 1 (lift-off); swing progress 0 (lift-off) →
// 1 (touchdown).
func TestLegProgress(t *testing.T) {
	tests := []struct {
		name     string
		pattern  Pattern
		phase    float32
		leg      int
		stance   bool
		progress float32
	}{
		{"tripod L1 mid stance", Tripod(), 0.25, HexL1, true, 0.5},
		{"tripod R1 mid swing", Tripod(), 0.25, HexR1, false, 0.5},
		{"tripod L1 touchdown", Tripod(), 0, HexL1, true, 0},
		{"tripod R1 lift-off", Tripod(), 0, HexR1, false, 0},
		{"tripod L1 mid swing", Tripod(), 0.75, HexL1, false, 0.5},
		{"walk LH", Walk(), 0.1, QuadLH, true, 0.1 / 0.75},
		{"walk RF", Walk(), 0.1, QuadRF, true, 0.35 / 0.75},
		{"walk LF swing", Walk(), 0.1, QuadLF, false, 0.4},
		{"walk RH", Walk(), 0.1, QuadRH, true, 0.6 / 0.75},
		{"wave L3 swing", Wave(), 0.9, HexL3, false, 0.4},
		{"wave R1 stance", Wave(), 0.9, HexR1, true, (0.9 - 5.0/6) / (5.0 / 6)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := mustNew(t, tc.pattern, 1)
			s.Reset(tc.phase)
			got := s.Leg(tc.leg)
			if got.Stance != tc.stance || math.Abs(float64(got.Progress-tc.progress)) > 1e-4 {
				t.Fatalf("leg %d at phase %v = %+v, want stance %v progress %.4f", tc.leg, tc.phase, got, tc.stance, tc.progress)
			}
			if got != s.State().Leg[tc.leg] {
				t.Fatalf("Leg() %+v != State().Leg %+v", got, s.State().Leg[tc.leg])
			}
		})
	}
}

func TestStanceAndSwingTime(t *testing.T) {
	tests := []struct {
		name          string
		pattern       Pattern
		period        float32
		stance, swing float32
	}{
		{"tripod", Tripod(), 2, 1, 1},
		{"wave", Wave(), 1.2, 1.0, 0.2},
		{"ripple", Ripple(), 0.9, 0.6, 0.3},
		{"walk", Walk(), 1.6, 1.2, 0.4},
		{"trot", Trot(), 0.5, 0.25, 0.25},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := mustNew(t, tc.pattern, tc.period)
			for leg := range tc.pattern.Legs {
				st, sw := s.StanceTime(leg), s.SwingTime(leg)
				if math.Abs(float64(st-tc.stance)) > 1e-5 || math.Abs(float64(sw-tc.swing)) > 1e-5 {
					t.Fatalf("leg %d stance/swing %v/%v, want %v/%v", leg, st, sw, tc.stance, tc.swing)
				}
			}
			if err := s.SetPeriod(2 * tc.period); err != nil {
				t.Fatal(err)
			}
			if got := s.StanceTime(0); math.Abs(float64(got-2*tc.stance)) > 1e-5 {
				t.Fatalf("after SetPeriod stance time %v, want %v", got, 2*tc.stance)
			}
		})
	}
}

func TestAdvanceDoesNotAllocate(t *testing.T) {
	for _, p := range []Pattern{Tripod(), Wave(), Ripple(), Walk(), Trot()} {
		s := mustNew(t, p, 1)
		var sink State
		allocs := testing.AllocsPerRun(1000, func() {
			s.Advance(0.02)
			sink = s.State()
			_ = s.Leg(0).PhaseState(0.2, 0.2)
		})
		if allocs != 0 {
			t.Fatalf("%s: %v allocs per Advance+State, want 0", p.Name, allocs)
		}
		_ = sink
	}
}

func TestPresetsValidate(t *testing.T) {
	for _, p := range []Pattern{Tripod(), Wave(), Ripple(), Walk(), Trot()} {
		if err := p.Validate(); err != nil {
			t.Fatalf("%s: %v", p.Name, err)
		}
	}
}

func TestNewRejectsInvalid(t *testing.T) {
	nan := float32(math.NaN())
	mod := func(f func(p *Pattern)) Pattern {
		p := Tripod()
		f(&p)
		return p
	}
	tests := []struct {
		name    string
		pattern Pattern
		period  float32
	}{
		{"no legs", mod(func(p *Pattern) { p.Legs = 0 }), 1},
		{"too many legs", mod(func(p *Pattern) { p.Legs = MaxLegs + 1 }), 1},
		{"duty 0", mod(func(p *Pattern) { p.Duty[2] = 0 }), 1},
		{"duty 1", mod(func(p *Pattern) { p.Duty[2] = 1 }), 1},
		{"duty NaN", mod(func(p *Pattern) { p.Duty[2] = nan }), 1},
		{"offset 1", mod(func(p *Pattern) { p.Offset[1] = 1 }), 1},
		{"offset negative", mod(func(p *Pattern) { p.Offset[1] = -0.1 }), 1},
		{"offset NaN", mod(func(p *Pattern) { p.Offset[1] = nan }), 1},
		{"period 0", Tripod(), 0},
		{"period negative", Tripod(), -1},
		{"period NaN", Tripod(), nan},
		{"period Inf", Tripod(), float32(math.Inf(1))},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(tc.pattern, tc.period); err == nil {
				t.Fatalf("New accepted %+v period %v", tc.pattern, tc.period)
			}
		})
	}
	s := mustNew(t, Tripod(), 1)
	if err := s.SetPeriod(0); err == nil {
		t.Fatal("SetPeriod(0) accepted")
	}
	if s.Period() != 1 {
		t.Fatalf("rejected SetPeriod changed period to %v", s.Period())
	}
}

// GaitTemplate.PhaseMap (touchdown offsets) and DutyCycles (stance fractions) drive
// the same scheduler as the equivalent preset.
func TestNewFromTemplate(t *testing.T) {
	roles := []string{"LF", "LH", "RF", "RH"}
	tmpl := gaittypes.GaitTemplate{
		ID:         "trot",
		CycleTime:  800 * time.Millisecond,
		PhaseMap:   map[string]float32{"LF": 0, "LH": 0.5, "RF": 0.5, "RH": 0},
		DutyCycles: map[string]float32{"LF": 0.5, "LH": 0.5, "RF": 0.5, "RH": 0.5},
	}
	got, err := NewFromTemplate(tmpl, roles)
	if err != nil {
		t.Fatal(err)
	}
	want := mustNew(t, Trot(), 0.8)
	if got.Period() != want.Period() {
		t.Fatalf("period %v, want %v", got.Period(), want.Period())
	}
	for k := range 1000 {
		ph := float32(k) / 1000
		got.Reset(ph)
		want.Reset(ph)
		if got.State() != want.State() {
			t.Fatalf("phase %v: template state %+v, preset state %+v", ph, got.State(), want.State())
		}
	}

	missing := tmpl
	missing.DutyCycles = map[string]float32{"LF": 0.5, "LH": 0.5, "RF": 0.5}
	if _, err := NewFromTemplate(missing, roles); err == nil {
		t.Fatal("accepted a template without a duty cycle for RH")
	}
	noOffset := tmpl
	noOffset.PhaseMap = map[string]float32{"LF": 0, "LH": 0.5, "RF": 0.5}
	if _, err := NewFromTemplate(noOffset, roles); err == nil {
		t.Fatal("accepted a template without a phase offset for RH")
	}
	noCycle := tmpl
	noCycle.CycleTime = 0
	if _, err := NewFromTemplate(noCycle, roles); err == nil {
		t.Fatal("accepted a zero cycle time")
	}
}

// LegState.PhaseState maps stance/swing progress onto the support planner's
// Support/Lift/Swing/Touchdown sub-phases.
func TestPhaseStateMapping(t *testing.T) {
	tests := []struct {
		name          string
		leg           LegState
		lift, landing float32
		mode          gaittypes.PhaseMode
		progress      float32
	}{
		{"stance", LegState{Stance: true, Progress: 0.4}, 0.2, 0.3, gaittypes.PhaseSupport, 0.4},
		{"lift", LegState{Progress: 0.1}, 0.2, 0.3, gaittypes.PhaseLift, 0.5},
		{"swing start", LegState{Progress: 0.2}, 0.2, 0.3, gaittypes.PhaseSwing, 0},
		{"swing mid", LegState{Progress: 0.45}, 0.2, 0.3, gaittypes.PhaseSwing, 0.5},
		{"touchdown start", LegState{Progress: 0.7}, 0.2, 0.3, gaittypes.PhaseTouchdown, 0},
		{"touchdown mid", LegState{Progress: 0.85}, 0.2, 0.3, gaittypes.PhaseTouchdown, 0.5},
		{"no lift segment", LegState{Progress: 0}, 0, 0.3, gaittypes.PhaseSwing, 0},
		{"fractions overlap: lift wins", LegState{Progress: 0.5}, 0.6, 0.6, gaittypes.PhaseLift, 0.5 / 0.6},
		{"fractions overlap: touchdown rest", LegState{Progress: 0.7}, 0.6, 0.6, gaittypes.PhaseTouchdown, 0.25},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.leg.PhaseState(tc.lift, tc.landing)
			if got.Mode != tc.mode || math.Abs(float64(got.Progress-tc.progress)) > 1e-6 {
				t.Fatalf("PhaseState = %+v, want mode %d progress %v", got, tc.mode, tc.progress)
			}
		})
	}
}
