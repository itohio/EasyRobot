package gaittypes

import (
	"context"
	"time"

	vec "github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// TerrainMode enumerates the level of terrain awareness available to planners.
type TerrainMode uint8

const (
	TerrainModeNone TerrainMode = iota
	TerrainModeHeights
	TerrainModeContactOnly
)

// PlannerFeature documents optional planner capabilities that callers can query.
type PlannerFeature string

const (
	FeaturePresetSwitching  PlannerFeature = "preset_switching"
	FeatureOrientationBlend PlannerFeature = "orientation_blend"
	FeatureTerrainAware     PlannerFeature = "terrain_aware"
	FeatureVariableTimestep PlannerFeature = "variable_timestep"
	FeaturePreview          PlannerFeature = "preview"
)

// FeatureSet captures the capabilities exposed by a planner implementation.
type FeatureSet map[PlannerFeature]bool

// PhaseMode enumerates the supported leg path sub-phases.
type PhaseMode uint8

const (
	PhaseUnknown PhaseMode = iota
	PhaseLift
	PhaseSwing
	PhaseTouchdown
	PhaseSupport
)

// PhaseState conveys the leg's current sub-phase and progress within that phase.
type PhaseState struct {
	Mode     PhaseMode
	Progress float32 // Normalized progress [0,1] within the current sub-phase.
}

// EndpointPose captures desired endpoint kinematics.
type EndpointPose struct {
	Position    vec.Vector3D
	Orientation *vec.Quaternion // Optional; nil keeps previous orientation.
}

// FootState represents the planner output for a leg endpoint.
type FootState struct {
	Pose          EndpointPose
	Velocity      vec.Vector3D
	Contact       bool
	Phase         PhaseMode
	PhaseProgress float32
	Timestamp     time.Time
}

// SupportObservations carries sensor feedback relevant to the support planner.
type SupportObservations struct {
	ContactDetected bool
	ContactHeight   *float32
}

// SupportUpdateRequest groups the inputs required for a support planner update.
type SupportUpdateRequest struct {
	LegID        string
	DesiredPose  EndpointPose
	Phase        PhaseState
	Timestamp    time.Time
	Delta        time.Duration
	Observations SupportObservations
}

// SupportProfile defines default clearances and behaviors for sub-phases.
type SupportProfile struct {
	LiftHeight         float32
	SwingHeight        float32
	TouchdownClearance float32
	SupportBlend       float32
}

// PathShape interpolates endpoint poses along a phase-specific curve.
type PathShape interface {
	Interpolate(start, target EndpointPose, t float32) EndpointPose
}

// Normalize coerces the profile values into valid ranges.
func (profile SupportProfile) Normalize() SupportProfile {
	if profile.LiftHeight < 0 {
		profile.LiftHeight = 0
	}
	if profile.SwingHeight < 0 {
		profile.SwingHeight = 0
	}
	if profile.SupportBlend < 0 {
		profile.SupportBlend = 0
	}
	if profile.SupportBlend > 1 {
		profile.SupportBlend = 1
	}
	return profile
}

// TerrainSample conveys surface data under the foot.
type TerrainSample struct {
	Position   vec.Vector3D
	Height     float32
	Normal     vec.Vector3D
	Confidence float32
	Contact    bool
	Timestamp  time.Time
}

// TerrainProvider supplies terrain feedback to planners.
type TerrainProvider interface {
	Mode() TerrainMode
	SampleFootprint(ctx context.Context, legID string, estimate vec.Vector3D) (TerrainSample, error)
}

// SupportEndpointPlanner defines the contract for support endpoint planners.
type SupportEndpointPlanner interface {
	Update(ctx context.Context, req SupportUpdateRequest) (FootState, error)
	Features() FeatureSet
}

// RigidBodyCommand encapsulates desired body motion.
type RigidBodyCommand struct {
	LinearVelocity  vec.Vector3D
	AngularVelocity vec.Vector3D
}

// RigidBodyState captures the pose estimate for the body.
type RigidBodyState struct {
	Position vec.Vector3D
	Rotation vec.Quaternion
	Twist    RigidBodyCommand
	Time     time.Time
}

// RigidBodyPlanner describes rigid body forward/backward kinematics contract.
type RigidBodyPlanner interface {
	Advance(ctx context.Context, cmd RigidBodyCommand, dt time.Duration) (RigidBodyState, error)
	EstimateFromEndpoints(ctx context.Context, feet map[string]FootState) (RigidBodyState, error)
	Features() FeatureSet
}

// GaitTemplate is a declarative description of gait timing.
type GaitTemplate struct {
	ID              string
	CycleTime       time.Duration
	PhaseMap        map[string]float32
	DutyCycles      map[string]float32
	ContactSequence []string
	SwingProfile    map[string]SupportProfile
	TransitionHints map[string]time.Duration
}

// LegRoleBinding maps template leg roles to physical leg identifiers.
type LegRoleBinding struct {
	Role    string
	LegIDs  []string
	Weights []float32
}

// GaitInstance is the runtime binding of a template to a robot.
type GaitInstance struct {
	Template   GaitTemplate
	Bindings   []LegRoleBinding
	Phase      float32
	Features   FeatureSet
	LastUpdate time.Time
}

// TransitionPolicy enumerates gait switch strategies.
type TransitionPolicy uint8

const (
	TransitionImmediate TransitionPolicy = iota
	TransitionCompleteCycle
	TransitionAtNextSupport
)

// GaitScheduler coordinates gait transitions.
type GaitScheduler interface {
	Active() GaitInstance
	SetTarget(templateID string, policy TransitionPolicy) error
	Tick(ctx context.Context, now time.Time) (GaitInstance, error)
}
