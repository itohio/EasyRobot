package rigidbody

import (
	"context"
	"errors"
	"sync"
	"time"

	gaittypes "github.com/itohio/EasyRobot/x/math/control/motion/gait/types"
	vec "github.com/itohio/EasyRobot/x/math/vec"
)

// ErrNotImplemented signals functionality that remains to be completed.
var ErrNotImplemented = errors.New("gait/rigidbody: not implemented")

// Option configures the Planner.
type Option func(*plannerOptions)

type plannerOptions struct {
	initialPose gaittypes.RigidBodyState
}

// WithInitialPose sets the starting pose for the planner.
func WithInitialPose(state gaittypes.RigidBodyState) Option {
	return func(o *plannerOptions) {
		o.initialPose = state
	}
}

// Planner maintains rigid body state and integrates motion commands.
type Planner struct {
	mu    sync.Mutex
	state gaittypes.RigidBodyState
	opts  plannerOptions
}

var _ gaittypes.RigidBodyPlanner = (*Planner)(nil)

// NewPlanner constructs a rigid body planner.
func NewPlanner(opts ...Option) *Planner {
	config := plannerOptions{}
	for _, opt := range opts {
		opt(&config)
	}

	return &Planner{
		state: config.initialPose,
		opts:  config,
	}
}

// Features enumerates optional behaviours supported by the planner.
func (p *Planner) Features() gaittypes.FeatureSet {
	return gaittypes.FeatureSet{
		gaittypes.FeatureVariableTimestep: true,
	}
}

// Advance integrates the body command forward by dt.
func (p *Planner) Advance(ctx context.Context, cmd gaittypes.RigidBodyCommand, dt time.Duration) (gaittypes.RigidBodyState, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if dt <= 0 {
		return gaittypes.RigidBodyState{}, errors.New("gait/rigidbody: dt must be positive")
	}

	// Placeholder implementation: mark twist and timestamp, but do not integrate pose.
	p.state.Twist = cmd
	p.state.Time = p.state.Time.Add(dt)
	return p.state, ErrNotImplemented
}

// EstimateFromEndpoints infers body pose from endpoint data.
func (p *Planner) EstimateFromEndpoints(ctx context.Context, feet map[string]gaittypes.FootState) (gaittypes.RigidBodyState, error) {
	return gaittypes.RigidBodyState{}, ErrNotImplemented
}

// Helpers (to be implemented) ------------------------------------------------

func integratePose(state gaittypes.RigidBodyState, cmd gaittypes.RigidBodyCommand, dt time.Duration) gaittypes.RigidBodyState {
	seconds := float32(dt.Seconds())
	next := state
	next.Position = vec.Vector3D{
		state.Position[0] + cmd.LinearVelocity[0]*seconds,
		state.Position[1] + cmd.LinearVelocity[1]*seconds,
		state.Position[2] + cmd.LinearVelocity[2]*seconds,
	}
	// TODO: integrate quaternion using angular velocity.
	return next
}
