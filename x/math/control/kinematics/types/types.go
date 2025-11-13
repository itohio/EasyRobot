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
)
