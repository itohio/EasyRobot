package support

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	gaittypes "github.com/itohio/EasyRobot/x/math/control/motion/gait/types"
	vec "github.com/itohio/EasyRobot/x/math/vec"
)

// Option configures the SupportPathPlanner.
type Option func(*supportOptions)

// WithTerrainProvider configures an optional terrain provider.
func WithTerrainProvider(provider gaittypes.TerrainProvider) Option {
	return func(o *supportOptions) {
		o.terrain = provider
	}
}

// WithSupportProfile overrides the default sub-phase profile.
func WithSupportProfile(profile gaittypes.SupportProfile) Option {
	return func(o *supportOptions) {
		o.profile = profile.Normalize()
	}
}

// WithPhaseShape overrides the interpolation shape for a specific phase.
func WithPhaseShape(phase gaittypes.PhaseMode, shape gaittypes.PathShape) Option {
	return func(o *supportOptions) {
		if shape == nil {
			return
		}
		if o.shapes == nil {
			o.shapes = make(map[gaittypes.PhaseMode]gaittypes.PathShape)
		}
		o.shapes[phase] = shape
	}
}

type supportOptions struct {
	terrain gaittypes.TerrainProvider
	profile gaittypes.SupportProfile
	shapes  map[gaittypes.PhaseMode]gaittypes.PathShape
}

// SupportPathPlanner produces time-dependent endpoint trajectories per leg.
type SupportPathPlanner struct {
	mu   sync.Mutex
	opts supportOptions
	legs map[string]*legState
}

// Ensure SupportPathPlanner satisfies the contract.
var _ gaittypes.SupportEndpointPlanner = (*SupportPathPlanner)(nil)

// NewSupportPathPlanner constructs a planner with optional configuration.
func NewSupportPathPlanner(opts ...Option) *SupportPathPlanner {
	defaultProfile := gaittypes.SupportProfile{
		LiftHeight:         0.05,
		SwingHeight:        0.08,
		TouchdownClearance: 0.0,
		SupportBlend:       1.0,
	}.Normalize()

	config := supportOptions{
		profile: defaultProfile,
	}

	for _, opt := range opts {
		opt(&config)
	}

	config.profile = config.profile.Normalize()
	config.shapes = applyDefaultShapes(config.shapes, config.profile)

	return &SupportPathPlanner{
		opts: config,
		legs: make(map[string]*legState),
	}
}

// Features enumerates the capabilities supported by this planner.
func (p *SupportPathPlanner) Features() gaittypes.FeatureSet {
	return gaittypes.FeatureSet{
		gaittypes.FeatureOrientationBlend: true,
		gaittypes.FeatureTerrainAware:     p.opts.terrain != nil,
		gaittypes.FeatureVariableTimestep: true,
	}
}

// Update advances the trajectory for the given leg and returns the new foot state.
func (p *SupportPathPlanner) Update(ctx context.Context, req gaittypes.SupportUpdateRequest) (gaittypes.FootState, error) {
	if err := validateSupportRequest(req); err != nil {
		return gaittypes.FootState{}, err
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	state := p.ensureLegState(req.LegID, req.DesiredPose)
	if err := p.syncSubPhase(ctx, state, req); err != nil {
		return gaittypes.FootState{}, err
	}

	progress := clamp01(req.Phase.Progress)
	nextPose := p.shapeForPhase(req.Phase.Mode).Interpolate(state.startPose, state.targetPose, progress)
	velocity, err := computeVelocity(state.currentPose.Pose.Position, nextPose.Position, req.Delta)
	if err != nil {
		return gaittypes.FootState{}, err
	}

	state.previousPose = state.currentPose
	state.currentPose = gaittypes.FootState{
		Pose:          nextPose,
		Velocity:      velocity,
		Contact:       determineContact(req.Phase.Mode, progress, req.Observations),
		Phase:         req.Phase.Mode,
		PhaseProgress: progress,
		Timestamp:     req.Timestamp,
	}
	state.lastPhase = req.Phase.Mode
	state.lastProgress = progress

	return state.currentPose, nil
}

func (p *SupportPathPlanner) ensureLegState(legID string, pose gaittypes.EndpointPose) *legState {
	state, ok := p.legs[legID]
	if ok {
		return state
	}

	initialState := gaittypes.FootState{
		Pose:      pose,
		Timestamp: time.Time{},
	}
	state = &legState{
		currentPose: initialState,
		startPose:   pose,
		targetPose:  pose,
	}
	p.legs[legID] = state
	return state
}

func (p *SupportPathPlanner) syncSubPhase(ctx context.Context, state *legState, req gaittypes.SupportUpdateRequest) error {
	phaseChanged := req.Phase.Mode != state.lastPhase
	progressWrapped := req.Phase.Progress < state.lastProgress
	if !phaseChanged && !progressWrapped {
		return nil
	}

	state.startPose = state.currentPose.Pose
	profile := p.opts.profile

	switch req.Phase.Mode {
	case gaittypes.PhaseLift:
		state.targetPose = liftTarget(state.currentPose.Pose, profile)
	case gaittypes.PhaseSwing:
		state.targetPose = swingTarget(state.currentPose.Pose, req.DesiredPose, profile)
	case gaittypes.PhaseTouchdown:
		target := touchdownTarget(state.currentPose.Pose, req.DesiredPose, profile)
		if err := p.resolveTouchdownTarget(ctx, req, &target); err != nil {
			return err
		}
		state.targetPose = target
		state.lastTouchdown = target
	case gaittypes.PhaseSupport:
		state.targetPose = supportTarget(state.currentPose.Pose, req.DesiredPose, profile, state.lastTouchdown)
	default:
		return fmt.Errorf("gait/support: unsupported phase mode %d", req.Phase.Mode)
	}

	state.lastPhase = req.Phase.Mode
	state.lastProgress = req.Phase.Progress
	return nil
}

func (p *SupportPathPlanner) resolveTouchdownTarget(ctx context.Context, req gaittypes.SupportUpdateRequest, target *gaittypes.EndpointPose) error {
	mode := gaittypes.TerrainModeNone
	if p.opts.terrain != nil {
		mode = p.opts.terrain.Mode()
	}

	switch mode {
	case gaittypes.TerrainModeNone:
		return nil
	case gaittypes.TerrainModeHeights:
		sample, err := p.sampleTerrain(ctx, req.LegID, target.Position)
		if err != nil {
			return err
		}
		target.Position[2] = sample.Height + p.opts.profile.TouchdownClearance
		return nil
	case gaittypes.TerrainModeContactOnly:
		if req.Observations.ContactHeight != nil {
			target.Position[2] = *req.Observations.ContactHeight + p.opts.profile.TouchdownClearance
			return nil
		}
		return nil
	default:
		return fmt.Errorf("gait/support: unknown terrain mode %d", mode)
	}
}

func (p *SupportPathPlanner) sampleTerrain(ctx context.Context, legID string, estimate vec.Vector3D) (gaittypes.TerrainSample, error) {
	if p.opts.terrain == nil {
		return gaittypes.TerrainSample{}, errors.New("terrain provider not configured")
	}
	return p.opts.terrain.SampleFootprint(ctx, legID, estimate)
}

type legState struct {
	currentPose   gaittypes.FootState
	previousPose  gaittypes.FootState
	startPose     gaittypes.EndpointPose
	targetPose    gaittypes.EndpointPose
	lastTouchdown gaittypes.EndpointPose
	lastPhase     gaittypes.PhaseMode
	lastProgress  float32
}

func liftTarget(current gaittypes.EndpointPose, profile gaittypes.SupportProfile) gaittypes.EndpointPose {
	target := clonePose(current)
	target.Position[2] = current.Position[2] + profile.LiftHeight
	return target
}

func swingTarget(current, desired gaittypes.EndpointPose, profile gaittypes.SupportProfile) gaittypes.EndpointPose {
	target := clonePose(desired)
	target.Position[2] = desired.Position[2] + profile.SwingHeight
	return target
}

func touchdownTarget(current, desired gaittypes.EndpointPose, profile gaittypes.SupportProfile) gaittypes.EndpointPose {
	target := clonePose(desired)
	target.Position[2] = desired.Position[2] + profile.TouchdownClearance
	return target
}

func supportTarget(current, desired gaittypes.EndpointPose, profile gaittypes.SupportProfile, touchdown gaittypes.EndpointPose) gaittypes.EndpointPose {
	target := clonePose(desired)
	if profile.SupportBlend < 1.0 {
		target.Position = interpolateVec3(touchdown.Position, desired.Position, profile.SupportBlend)
	}
	return target
}

func (p *SupportPathPlanner) shapeForPhase(mode gaittypes.PhaseMode) gaittypes.PathShape {
	if shape, ok := p.opts.shapes[mode]; ok {
		return shape
	}
	return linearShape{}
}

func determineContact(mode gaittypes.PhaseMode, progress float32, obs gaittypes.SupportObservations) bool {
	switch mode {
	case gaittypes.PhaseTouchdown:
		if obs.ContactDetected {
			return true
		}
		return progress >= 0.5
	case gaittypes.PhaseSupport:
		return true
	default:
		return false
	}
}

func computeVelocity(previous, next vec.Vector3D, delta time.Duration) (vec.Vector3D, error) {
	if delta <= 0 {
		return vec.Vector3D{}, fmt.Errorf("gait/support: delta must be positive, got %s", delta)
	}
	seconds := float32(delta.Seconds())
	return vec.Vector3D{
		(next[0] - previous[0]) / seconds,
		(next[1] - previous[1]) / seconds,
		(next[2] - previous[2]) / seconds,
	}, nil
}

func interpolateVec3(a, b vec.Vector3D, t float32) vec.Vector3D {
	return vec.Vector3D{
		a[0] + (b[0]-a[0])*t,
		a[1] + (b[1]-a[1])*t,
		a[2] + (b[2]-a[2])*t,
	}
}

func interpolateQuat(a, b *vec.Quaternion, t float32) *vec.Quaternion {
	switch {
	case a == nil && b == nil:
		return nil
	case a == nil:
		return cloneQuat(b)
	case b == nil:
		return cloneQuat(a)
	default:
		qa := *a
		qb := *b
		interpolated := qa.Slerp(qb, t, 0).(vec.Quaternion)
		return &interpolated
	}
}

func clonePose(p gaittypes.EndpointPose) gaittypes.EndpointPose {
	return gaittypes.EndpointPose{
		Position:    p.Position,
		Orientation: cloneQuat(p.Orientation),
	}
}

func cloneQuat(q *vec.Quaternion) *vec.Quaternion {
	if q == nil {
		return nil
	}
	copy := *q
	return &copy
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func validateSupportRequest(req gaittypes.SupportUpdateRequest) error {
	if req.LegID == "" {
		return errors.New("gait/support: leg id required")
	}
	if req.Delta <= 0 {
		return fmt.Errorf("gait/support: delta must be positive, got %s", req.Delta)
	}
	if req.Phase.Mode == gaittypes.PhaseUnknown {
		return errors.New("gait/support: phase mode required")
	}
	if req.Phase.Progress < 0 || req.Phase.Progress > 1 {
		return fmt.Errorf("gait/support: phase progress out of range [0,1]: %.3f", req.Phase.Progress)
	}
	return nil
}

// Path shape implementations --------------------------------------------------

type linearShape struct{}

func (linearShape) Interpolate(start, target gaittypes.EndpointPose, t float32) gaittypes.EndpointPose {
	return gaittypes.EndpointPose{
		Position:    interpolateVec3(start.Position, target.Position, t),
		Orientation: interpolateQuat(start.Orientation, target.Orientation, t),
	}
}

type arcShape struct {
	amplitude float32
}

func (s arcShape) Interpolate(start, target gaittypes.EndpointPose, t float32) gaittypes.EndpointPose {
	pos := interpolateVec3(start.Position, target.Position, t)
	z0 := start.Position[2]
	z1 := target.Position[2]
	apex := maxFloat32(z0, z1) + s.amplitude
	pos[2] = quadraticThroughMidpoint(z0, apex, z1, t)

	return gaittypes.EndpointPose{
		Position:    pos,
		Orientation: interpolateQuat(start.Orientation, target.Orientation, t),
	}
}

func quadraticThroughMidpoint(z0, apex, z1, t float32) float32 {
	dz := z1 - z0
	b := 4*(apex-z0) - dz
	a := 2*dz - 4*(apex-z0)
	return a*t*t + b*t + z0
}

func defaultPhaseShapes(profile gaittypes.SupportProfile) map[gaittypes.PhaseMode]gaittypes.PathShape {
	amp := defaultArcAmplitude(profile)
	return map[gaittypes.PhaseMode]gaittypes.PathShape{
		gaittypes.PhaseSupport:   linearShape{},
		gaittypes.PhaseLift:      arcShape{amplitude: amp},
		gaittypes.PhaseSwing:     arcShape{amplitude: amp},
		gaittypes.PhaseTouchdown: arcShape{amplitude: amp},
	}
}

func applyDefaultShapes(existing map[gaittypes.PhaseMode]gaittypes.PathShape, profile gaittypes.SupportProfile) map[gaittypes.PhaseMode]gaittypes.PathShape {
	defaults := defaultPhaseShapes(profile)
	if existing == nil {
		return defaults
	}

	for mode, shape := range defaults {
		if _, ok := existing[mode]; !ok {
			existing[mode] = shape
		}
	}
	return existing
}

func defaultArcAmplitude(profile gaittypes.SupportProfile) float32 {
	maxHeight := maxFloat32(profile.LiftHeight, profile.SwingHeight)
	return maxFloat32(maxHeight, profile.TouchdownClearance)
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
