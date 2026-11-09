package types

import (
	"errors"

	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
)

// Config groups immutable configuration parameters required to instantiate a
// kinematic model (geometry, mass properties, actuator layout, etc.).
type Config struct {
	Name             string
	DegreesOfFreedom int
	ActuatorCount    int
	StateDimension   int
	ControlDimension int
	Mass             float32
	Metadata         map[string]float32
}

// Dimensions describes the state/control dimensionality exposed by a model.
type Dimensions struct {
	StateRows    int
	StateCols    int
	ControlSize  int
	ActuatorSize int
}

// Capabilities encodes coarse-grained kinematic properties consumers can use
// to select appropriate planners or constraint handlers.
type Capabilities struct {
	Holonomic        bool
	Omnidirectional  bool
	Underactuated    bool
	SupportsLateral  bool
	SupportsVertical bool
	ConstraintRank   int
}

// Constraints captures common constraint sets that models expose, expressed
// as symmetric bounds on states and controls and optional coupling matrices.
type Constraints struct {
	StateLower   mattype.Matrix
	StateUpper   mattype.Matrix
	StateRate    mattype.Matrix
	ControlLower mattype.Matrix
	ControlUpper mattype.Matrix
	ControlRate  mattype.Matrix
	Coupling     mattype.Matrix
}

// Model represents the minimal surface area shared by all kinematics models.
type Model interface {
	Dimensions() Dimensions
	Capabilities() Capabilities
	ConstraintSet() Constraints
}

// ForwardKinematics defines the destination-based forward propagation contract.
type ForwardKinematics interface {
	Model
	Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
}

// BackwardKinematics defines the destination-based inverse mapping contract.
type BackwardKinematics interface {
	Model
	Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
}

// Bidirectional couples forward and backward capabilities.
type Bidirectional interface {
	ForwardKinematics
	BackwardKinematics
}

var (
	// ErrInvalidDimensions is returned when matrices or vectors do not match
	// the dimensional requirements advertised by a model.
	ErrInvalidDimensions = errors.New("kinematics/types: invalid dimensions")

	// ErrUnsupportedOperation indicates the model cannot satisfy the requested
	// operation with the provided capabilities.
	ErrUnsupportedOperation = errors.New("kinematics/types: unsupported operation")

	// ErrInvalidConfig is returned by constructors and Validate methods when a
	// model or joint configuration is malformed (non-finite values, Min >= Max,
	// unsupported joint kinds, …). Validation errors wrap it.
	ErrInvalidConfig = errors.New("kinematics/types: invalid configuration")

	// ErrNonFinite is returned when an input (target, seed state) contains NaN
	// or ±Inf. Solvers reject such input before solving and leave outputs
	// untouched.
	ErrNonFinite = errors.New("kinematics/types: non-finite input")

	// ErrUnreachable indicates an inverse-kinematics target lies outside the
	// workspace of the mechanism. Solvers still write a best-effort solution.
	ErrUnreachable = errors.New("kinematics/types: target unreachable")

	// ErrJointLimit indicates an inverse-kinematics target is geometrically
	// reachable but only with a joint beyond its limits. Solvers still write a
	// best-effort (limit-respecting) solution.
	ErrJointLimit = errors.New("kinematics/types: joint limit")

	// ErrNoConvergence indicates an iterative inverse-kinematics solver ran out
	// of its iteration budget (or stalled) without a typed reason. Solvers still
	// write a best-effort solution.
	ErrNoConvergence = errors.New("kinematics/types: inverse kinematics did not converge")
)
