package scheduler

import "testing"

// Review R-ER-017-018 G9 (S1): offsets are rounded to the nearest 2⁻²⁰ grid
// point, not truncated: an offset within 0.4 grid units of k·2⁻²⁰ touches
// down exactly at phase k·2⁻²⁰ and is still in swing one progress quantum
// (2⁻²⁴) earlier.
func TestOffsetRoundsToGrid(t *testing.T) {
	const k = 1<<18 + 3
	at := float32(k) / grid
	for _, d := range []float64{-0.4, -0.1, 0, 0.1, 0.4} {
		p := Pattern{Name: "one", Legs: 1}
		p.Offset[0] = float32((k + d) / grid)
		p.Duty[0] = 0.5
		s := mustNew(t, p, 1)
		s.Reset(at)
		if got := s.Leg(0); !got.Stance || got.Progress != 0 {
			t.Fatalf("offset k%+.1f grid: at phase k·2⁻²⁰ leg is %+v, want touchdown (stance, progress 0)", d, got)
		}
		s.Reset(at - 1.0/(1<<24))
		if got := s.Leg(0); got.Stance {
			t.Fatalf("offset k%+.1f grid: 2⁻²⁴ before phase k·2⁻²⁰ leg is %+v, want swing", d, got)
		}
	}
}

// Review G8: Advance ignores dt >= 2²⁴·period (no fractional cycle is left
// in float32); a shorter dt moves the phase by its fractional cycles.
func TestAdvanceIgnoresHugeDt(t *testing.T) {
	const period = 0.5
	s := mustNew(t, Tripod(), period)
	s.Reset(0.25)
	s.Advance((1 << 24) * period)
	if s.Phase() != 0.25 {
		t.Fatalf("Advance(2²⁴·period) moved the phase to %v", s.Phase())
	}
	s.Advance((1<<20 + 0.5) * period)
	if s.Phase() != 0.75 {
		t.Fatalf("Advance((2²⁰ + ½)·period) = phase %v, want 0.75", s.Phase())
	}
}
