package locomotion

import (
	"time"

	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// Locomotion represents any mobile robot locomotion system
type Locomotion interface {
	// Update processes sensor readings and computes actuator commands
	Update(sensorReadings SensorData, dt time.Duration) error

	// SetTarget sets locomotion targets (velocity, position, orientation)
	SetTarget(target Target) error

	// GetState returns current locomotion state
	GetState() State

	// Configure sets locomotion parameters
	Configure(config Config) error
}

// Appendage represents a kinematic chain (arm, leg) using DH parameters
type Appendage interface {
	// Forward computes end-effector pose from joint parameters
	Forward(jointAngles, effectorPose vec.Vector) error

	// Inverse computes joint angles from target end-effector pose
	Inverse(targetPose, jointAngles vec.Vector) error

	// SetConstraints updates joint and workspace constraints
	SetConstraints(constraints ConstraintConfig) error

	// DOF returns degrees of freedom
	DOF() int
}

// Drive represents wheeled locomotion systems
type Drive interface {
	// UpdateWheelSpeeds computes wheel speeds from body velocity commands
	UpdateWheelSpeeds(bodyVelocity, wheelSpeeds vec.Vector) error

	// UpdateBodyVelocity computes body velocity from wheel encoder readings
	UpdateBodyVelocity(encoderTicks vec.Vector, dt time.Duration, bodyVelocity vec.Vector) error

	// DeadReckoning updates position estimate using odometry
	DeadReckoning(encoderTicks vec.Vector, dt time.Duration, position, orientation vec.Vector) error
}

// SensorData contains all sensor readings for locomotion
type SensorData struct {
	EncoderTicks []int32    // Wheel encoder readings (ticks)
	JointAngles  vec.Vector // Current joint angles (radians)
	IMU          IMUData    // Orientation/acceleration data
	TouchSensors []bool     // Foot/ground contact sensors
	ForceSensors vec.Vector // Force/torque sensor readings
}

// IMUData represents inertial measurement unit readings
type IMUData struct {
	Acceleration  vec.Vector3D // Linear acceleration (m/s²)
	AngularVel    vec.Vector3D // Angular velocity (rad/s)
	MagneticField vec.Vector3D // Magnetic field (µT)
	Temperature   float32      // Temperature (°C)
}

// Target represents desired locomotion targets
type Target struct {
	LinearVelocity  vec.Vector3D // Desired linear velocity [vx, vy, vz] (m/s)
	AngularVelocity vec.Vector3D // Desired angular velocity [ωx, ωy, ωz] (rad/s)
	Position        vec.Vector3D // Target position [x, y, z] (m)
	Orientation     vec.Vector   // Target orientation (quaternion [w, x, y, z])
	BodyPose        BodyPose     // For legged robots - body pose control
}

// BodyPose represents body pose for legged robots
type BodyPose struct {
	Height float32 // Body height above ground (m)
	Roll   float32 // Body roll angle (rad)
	Pitch  float32 // Body pitch angle (rad)
	Yaw    float32 // Body yaw angle (rad)
}

// State represents current locomotion state
type State struct {
	Position     vec.Vector3D // Current position estimate [x, y, z] (m)
	Orientation  vec.Vector   // Current orientation (quaternion [w, x, y, z])
	Velocity     vec.Vector3D // Current velocity [vx, vy, vz] (m/s)
	JointAngles  vec.Vector   // Current joint angles (radians)
	ContactState []bool       // Leg/foot contact states
	Stability    float32      // Stability metric (0-1, higher is more stable)
	Error        error        // Last error encountered
}

// Config is the base configuration interface
type Config interface {
	Validate() error
}

// ConstraintConfig defines joint and workspace constraints
type ConstraintConfig struct {
	JointLimits     []vec.Vector     // [min, max] for each joint (radians or meters)
	Workspace       ConstraintVolume // Valid end-effector workspace
	MaxVelocity     vec.Vector       // Maximum joint velocities
	MaxAcceleration vec.Vector       // Maximum joint accelerations
}

// ConstraintVolume defines valid workspace constraints
type ConstraintVolume struct {
	MinPosition vec.Vector3D // Minimum position [x, y, z]
	MaxPosition vec.Vector3D // Maximum position [x, y, z]
	MaxReach    float32      // Maximum reach from origin
}

// GaitConfig defines gait pattern parameters
type GaitConfig struct {
	Type         GaitType  // Type of gait pattern
	Frequency    float32   // Gait frequency (Hz)
	DutyFactor   float32   // Duty factor (0-1, fraction of cycle in stance)
	StepHeight   float32   // Maximum foot lift height (m)
	StepLength   float32   // Step length (m)
	PhaseOffsets []float32 // Phase offsets for each leg (radians)
}

// GaitType defines different gait patterns
type GaitType int

const (
	GaitWalk   GaitType = iota // Sequential leg lifting (4-beat)
	GaitTrot                   // Diagonal leg pairs (2-beat)
	GaitGallop                 // Three legs support, one flight (4-beat asymmetric)
	GaitTripod                 // Three legs up, three down (hexapod)
	GaitWave                   // Sequential leg lifting (smooth)
	GaitRipple                 // Two adjacent legs up (hexapod)
)

// StabilizationConfig defines balance control parameters
type StabilizationConfig struct {
	EnableCOMControl   bool    // Enable center of mass control
	COMHeight          float32 // Target center of mass height
	MaxRollCorrection  float32 // Maximum roll correction (rad)
	MaxPitchCorrection float32 // Maximum pitch correction (rad)
	StabilityThreshold float32 // Minimum stability margin
	CorrectionGain     float32 // PID gain for pose correction
}

// ControlMode defines control modes for manipulators
type ControlMode int

const (
	ControlJoint      ControlMode = iota // Direct joint angle control
	ControlCartesian                     // End-effector pose control
	ControlTrajectory                    // Smooth trajectory following
)

// Trajectory defines a motion trajectory
type Trajectory struct {
	Waypoints     []vec.Vector    // Sequence of target poses/positions
	Velocities    []vec.Vector    // Target velocities at each waypoint
	Accelerations []vec.Vector    // Target accelerations at each waypoint
	Timestamps    []time.Duration // Time to reach each waypoint
	Mode          TrajectoryMode  // Interpolation mode
}

// TrajectoryMode defines trajectory interpolation modes
type TrajectoryMode int

const (
	TrajectoryLinear   TrajectoryMode = iota // Linear interpolation
	TrajectoryCubic                          // Cubic spline interpolation
	TrajectoryCircular                       // Circular arc interpolation
)
