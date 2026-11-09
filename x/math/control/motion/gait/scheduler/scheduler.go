// Package scheduler is the gait phase scheduler: one global gait phase φ ∈ [0,1)
// advanced by dt/period, a per-leg phase offset and duty factor β, and per-leg
// stance/swing state with progress.
//
// Timing rule: leg i is in stance when ψᵢ = (φ − offsetᵢ) mod 1 < βᵢ. The offset is
// the phase at which the leg touches down; the leg lifts off at offset + β.
//
//	stance progress = ψ/β            0 at touchdown → 1 at lift-off
//	swing progress  = (ψ − β)/(1 − β) 0 at lift-off → 1 at touchdown
//
// Leg indexing: the left side front to back, then the right side front to back.
// Hexapod: 0 L1 (left front), 1 L2 (left middle), 2 L3 (left rear), 3 R1, 4 R2, 5 R3.
// Quadruped: 0 LF, 1 LH, 2 RF, 3 RH. The constants HexL1… and QuadLF… name them.
//
// The phase is held as a uint32 fixed-point fraction of a cycle, so wrap-around and
// the modulo are exact. Offsets and swing lengths are quantised to 2⁻²⁰ of a cycle
// (offsets rounded, swing lengths rounded down): a float32 1/6 or 1/3 then lands on
// the same grid point as its exact value, and legs that hand over support at the
// same instant (tripod groups, wave neighbours) never overlap in swing. Advance and
// State do not allocate.
package scheduler

import (
	"errors"
	"fmt"
	"math"

	gaittypes "github.com/itohio/EasyRobot/x/math/control/motion/gait/types"
)

// MaxLegs is the largest number of legs a Pattern can describe.
const MaxLegs = 8

// Hexapod leg indices.
const (
	HexL1 = iota // left front
	HexL2        // left middle
	HexL3        // left rear
	HexR1        // right front
	HexR2        // right middle
	HexR3        // right rear
)

// Quadruped leg indices.
const (
	QuadLF = iota // left front
	QuadLH        // left hind
	QuadRF        // right front
	QuadRH        // right hind
)

const (
	gridBits  = 20
	grid      = 1 << gridBits
	gridShift = 32 - gridBits
	// progressShift drops the low phase bits so that values fit a float32 mantissa
	// exactly; it is below gridShift, so stance/swing lengths stay exact.
	progressShift = 8
	cycleTicks    = 1 << 32
)

var (
	// ErrInvalidPattern reports a pattern with a bad leg count, offset or duty factor.
	ErrInvalidPattern = errors.New("gait/scheduler: invalid pattern")
	// ErrInvalidPeriod reports a non-positive or non-finite cycle period.
	ErrInvalidPeriod = errors.New("gait/scheduler: period must be positive and finite")
)

// Pattern describes gait timing: per-leg touchdown phase offsets in [0,1) and duty
// factors β ∈ (0,1) (stance fraction of the cycle). Legs follow the package
// indexing convention.
type Pattern struct {
	Name   string
	Legs   int
	Offset [MaxLegs]float32
	Duty   [MaxLegs]float32
}

func uniform(name string, duty float32, offsets ...float32) Pattern {
	p := Pattern{Name: name, Legs: len(offsets)}
	for i, o := range offsets {
		p.Offset[i] = o
		p.Duty[i] = duty
	}
	return p
}

// Tripod is the hexapod tripod gait: β = 1/2, L1 R2 L3 touch down at 0 and R1 L2 R3
// at 1/2; always exactly three legs in stance.
func Tripod() Pattern {
	return uniform("tripod", 0.5, 0, 0.5, 0, 0.5, 0, 0.5)
}

// Ripple is the hexapod ripple gait: β = 2/3, each side a back-to-front wave with
// legs 1/3 apart, the right side 1/2 behind the left; one leg per side in swing.
func Ripple() Pattern {
	return uniform("ripple", 2.0/3,
		2.0/3, 1.0/3, 0, // L1 L2 L3
		1.0/6, 5.0/6, 0.5) // R1 R2 R3
}

// Wave is the hexapod wave gait: β = 5/6, one leg in swing at a time, back to front
// on the left side (L3 L2 L1), then back to front on the right side (R3 R2 R1).
func Wave() Pattern {
	return uniform("wave", 5.0/6,
		2.0/6, 1.0/6, 0, // L1 L2 L3
		5.0/6, 4.0/6, 3.0/6) // R1 R2 R3
}

// Walk is the quadruped lateral-sequence walk: β = 3/4, swings LH → LF → RH → RF,
// one leg in swing at a time.
func Walk() Pattern {
	return uniform("walk", 0.75,
		0.25, 0, // LF LH
		0.75, 0.5) // RF RH
}

// Trot is the quadruped trot: β = 1/2, diagonal pairs LF+RH and RF+LH half a cycle
// apart. Not statically stable.
func Trot() Pattern {
	return uniform("trot", 0.5,
		0, 0.5, // LF LH
		0.5, 0) // RF RH
}

// Validate checks the leg count, offsets ∈ [0,1) and duty factors ∈ (0,1) with a
// swing and a stance of at least 2⁻²⁰ of a cycle.
func (p Pattern) Validate() error {
	if p.Legs < 1 || p.Legs > MaxLegs {
		return fmt.Errorf("%w: %d legs, want 1..%d", ErrInvalidPattern, p.Legs, MaxLegs)
	}
	for i := range p.Legs {
		if o := p.Offset[i]; !(o >= 0 && o < 1) {
			return fmt.Errorf("%w: leg %d offset %v outside [0,1)", ErrInvalidPattern, i, o)
		}
		if sw := swingGrid(p.Duty[i]); !(sw >= 1 && sw < grid) {
			return fmt.Errorf("%w: leg %d duty %v outside (0,1)", ErrInvalidPattern, i, p.Duty[i])
		}
	}
	return nil
}

// swingGrid is the swing length in grid units, rounded down; NaN maps to 0.
func swingGrid(duty float32) float64 {
	sw := math.Floor((1 - float64(duty)) * grid)
	if sw != sw {
		return 0
	}
	return sw
}

// LegState is one leg's contact state. Progress is stance progress (0 touchdown →
// 1 lift-off) when Stance, otherwise swing progress (0 lift-off → 1 touchdown);
// always in [0,1).
type LegState struct {
	Stance   bool
	Progress float32
}

// PhaseState maps the leg state onto the support planner's sub-phases: stance is
// PhaseSupport; the swing is split into PhaseLift for the first liftFrac of the
// swing, PhaseTouchdown for the last touchdownFrac and PhaseSwing in between, each
// with its own progress in [0,1). liftFrac is clamped to [0,1] and touchdownFrac to
// [0, 1−liftFrac].
func (l LegState) PhaseState(liftFrac, touchdownFrac float32) gaittypes.PhaseState {
	if l.Stance {
		return gaittypes.PhaseState{Mode: gaittypes.PhaseSupport, Progress: l.Progress}
	}
	liftFrac = min(max(liftFrac, 0), 1)
	touchdownFrac = min(max(touchdownFrac, 0), 1-liftFrac)
	s := l.Progress
	landing := 1 - touchdownFrac
	switch {
	case s < liftFrac:
		return gaittypes.PhaseState{Mode: gaittypes.PhaseLift, Progress: s / liftFrac}
	case s < landing:
		return gaittypes.PhaseState{Mode: gaittypes.PhaseSwing, Progress: (s - liftFrac) / (landing - liftFrac)}
	default:
		return gaittypes.PhaseState{Mode: gaittypes.PhaseTouchdown, Progress: (s - landing) / touchdownFrac}
	}
}

// State is the scheduler output for one tick.
type State struct {
	Phase float32 // global gait phase φ ∈ [0,1)
	Legs  int     // number of valid entries in Leg
	Leg   [MaxLegs]LegState
}

// StanceCount returns the number of legs in stance.
func (s State) StanceCount() int {
	n := 0
	for i := range s.Legs {
		if s.Leg[i].Stance {
			n++
		}
	}
	return n
}

// Scheduler advances the gait phase and reports per-leg stance/swing state.
type Scheduler struct {
	pattern Pattern
	period  float32
	phase   uint32
	offset  [MaxLegs]uint32
	stance  [MaxLegs]uint32 // β in phase ticks
	swing   [MaxLegs]uint32 // 1 − β in phase ticks
	state   State
}

// New returns a scheduler for pattern p with cycle period in seconds, at phase 0.
func New(p Pattern, period float32) (*Scheduler, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if !validPeriod(period) {
		return nil, ErrInvalidPeriod
	}
	s := &Scheduler{pattern: p, period: period}
	for i := range p.Legs {
		off := uint32(math.Round(float64(p.Offset[i])*grid)) & (grid - 1)
		sw := uint32(swingGrid(p.Duty[i]))
		s.offset[i] = off << gridShift
		s.swing[i] = sw << gridShift
		s.stance[i] = (grid - sw) << gridShift
	}
	s.state.Legs = p.Legs
	s.update()
	return s, nil
}

// NewFromTemplate builds a scheduler from a GaitTemplate: legs lists the template
// roles in leg-index order; PhaseMap[role] is the touchdown offset, DutyCycles[role]
// the duty factor and CycleTime the period. Every role needs both entries.
func NewFromTemplate(t gaittypes.GaitTemplate, legs []string) (*Scheduler, error) {
	if len(legs) > MaxLegs {
		return nil, fmt.Errorf("%w: %d legs, want 1..%d", ErrInvalidPattern, len(legs), MaxLegs)
	}
	p := Pattern{Name: t.ID, Legs: len(legs)}
	for i, role := range legs {
		off, ok := t.PhaseMap[role]
		if !ok {
			return nil, fmt.Errorf("%w: template %q has no phase offset for %q", ErrInvalidPattern, t.ID, role)
		}
		duty, ok := t.DutyCycles[role]
		if !ok {
			return nil, fmt.Errorf("%w: template %q has no duty cycle for %q", ErrInvalidPattern, t.ID, role)
		}
		p.Offset[i] = off
		p.Duty[i] = duty
	}
	return New(p, float32(t.CycleTime.Seconds()))
}

func validPeriod(period float32) bool {
	return period > 0 && period <= math.MaxFloat32
}

// Advance moves the phase forward by dt seconds and updates the leg states.
// Non-positive and non-finite dt are ignored, and so is dt >= 2²⁴·period
// (float32 keeps no fraction of a cycle there); below that only the
// fractional cycles of dt/period count. Callers that can stall for that long
// should Reset instead.
func (s *Scheduler) Advance(dt float32) {
	if !(dt > 0) {
		return
	}
	x := dt / s.period
	if x >= 1 {
		if x >= 1<<24 { // no fractional part left in float32 (also +Inf)
			return
		}
		x -= float32(int32(x))
	}
	// x·2³² is exact in float32 (it only rescales the exponent); truncation to
	// uint32 is the modulo.
	s.phase += uint32(uint64(x * cycleTicks))
	s.update()
}

func (s *Scheduler) update() {
	for i := range s.pattern.Legs {
		psi := s.phase - s.offset[i]
		if psi < s.stance[i] {
			s.state.Leg[i] = LegState{
				Stance:   true,
				Progress: float32(psi>>progressShift) / float32(s.stance[i]>>progressShift),
			}
		} else {
			s.state.Leg[i] = LegState{
				Progress: float32((psi-s.stance[i])>>progressShift) / float32(s.swing[i]>>progressShift),
			}
		}
	}
	s.state.Phase = phaseToFloat(s.phase)
}

func phaseToFloat(phase uint32) float32 {
	return float32(phase>>progressShift) / (cycleTicks >> progressShift)
}

// State returns the current phase and leg states.
func (s *Scheduler) State() State { return s.state }

// Leg returns the state of leg i; i must be in [0, Pattern().Legs).
func (s *Scheduler) Leg(i int) LegState { return s.state.Leg[i] }

// Phase returns the global gait phase φ ∈ [0,1).
func (s *Scheduler) Phase() float32 { return s.state.Phase }

// Period returns the cycle period in seconds.
func (s *Scheduler) Period() float32 { return s.period }

// SetPeriod changes the cycle period (seconds) without moving the phase.
func (s *Scheduler) SetPeriod(period float32) error {
	if !validPeriod(period) {
		return ErrInvalidPeriod
	}
	s.period = period
	return nil
}

// Reset sets the global phase to the fractional part of phase.
func (s *Scheduler) Reset(phase float32) {
	f := float64(phase)
	f -= math.Floor(f)
	if f != f { // NaN, ±Inf
		f = 0
	}
	s.phase = uint32(uint64(f * cycleTicks))
	s.update()
}

// Pattern returns the pattern the scheduler was built with.
func (s *Scheduler) Pattern() Pattern { return s.pattern }

// StanceTime returns leg i's stance duration in seconds (β·period, β as quantised).
func (s *Scheduler) StanceTime(i int) float32 {
	return phaseToFloat(s.stance[i]) * s.period
}

// SwingTime returns leg i's swing duration in seconds ((1−β)·period, quantised).
func (s *Scheduler) SwingTime(i int) float32 {
	return phaseToFloat(s.swing[i]) * s.period
}
