# Proto Definitions

This directory contains Protocol Buffer definitions for the EasyRobot project. The definitions are organized into logical modules for easy navigation and maintenance.

## Directory Structure

```
proto/
├── types/
│   ├── math/              # Mathematical types
│   │   ├── vectors.proto   # Vector types (2D, 3D, 4D, Quaternion, generic)
│   │   ├── matrices.proto # Matrix types (2x2, 3x3, 4x4, 3x4, 4x3, generic)
│   │   └── tensor.proto    # Tensor and DataType definitions
│   ├── nn/                 # Neural network types
│   │   └── neural_network.proto  # Parameter, Layer, Model definitions
│   ├── control/            # Control and motion types
│   │   ├── path.proto      # Path with position and quaternion orientation
│   │   ├── motion.proto    # Motion planning (State, Controls, Trajectory, Constraints, PID)
│   │   └── kinematics/    # Kinematics organized by type
│   │       ├── joints.proto      # Joint-based kinematics (DH, planar)
│   │       ├── wheels.proto      # Wheel-based kinematics (differential, mecanum, steering)
│   │       ├── rigidbody.proto   # Rigid body kinematics
│   │       ├── types.proto       # Common kinematics types (Config, Dimensions, Capabilities)
│   │       └── state.proto       # Kinematics state, sensors, targets
│   ├── filter/             # Filter types
│   │   └── slam.proto      # SLAM data structures (OccupancyGrid, SLAMPose, SLAMConfig)
│   ├── interaction/        # User interaction types
│   │   ├── events.proto    # Mouse and keyboard events
│   │   ├── messages.proto  # User messages and commands (text form)
│   │   └── requests.proto  # User requests/responses with Any payload
│   └── core/               # Core types
│       ├── frame.proto     # Frame with index, timestamp, metadata, tensors
│       └── arbitrary.proto # Arbitrary type wrapper for local transfer
└── buf.yaml                # Buf configuration
```

## Package Organization

All proto files use package names that match their directory structure:
- `types.math` - Mathematical types
- `types.nn` - Neural network types
- `types.control` - Control and motion types
- `types.control.kinematics` - Kinematics types (organized by implementation)
- `types.filter` - Filter types (SLAM, etc.)
- `types.interaction` - User interaction types (events, messages, requests)
- `types.core` - Core types

## Key Types

### Math Types (`types/math/`)

- **Vectors**: `Vector2D`, `Vector3D`, `Vector4D`, `Quaternion`, `Vector` (generic)
- **Matrices**: `Matrix2x2`, `Matrix3x3`, `Matrix4x4`, `Matrix3x4`, `Matrix4x3`, `Matrix` (generic)
- **Tensors**: `Tensor` with `DataType` enum supporting FP32, FP64, INT8-64, UINT8, etc.

### Neural Networks (`types/nn/`)

- **Parameter**: Trainable parameter with data, gradient, and requires_grad flag
- **Layer**: Neural network layer with name, type, shapes, parameters, and config
- **Model**: Neural network model with layers, parameters, and metadata

### Control Types (`types/control/`)

- **Path**: `Path` with position (Vector3D) and orientation (Quaternion), `PathSegment`, `PathPlan`
- **Motion**: `State`, `Controls`, `Trajectory`, `Constraints`, `PIDGains`, `Parameters`

### Kinematics (`types/control/kinematics/`)

Organized to match `x/math/control/kinematics` structure:

- **Joints** (`joints.proto`): Joint-based kinematics
  - `DenavitHartenberg` - DH parameters
  - `PlanarJoint` - Planar joint configuration
  - `JointConstraint` - Joint limits and constraints
  - `JointsConfig` - Joint-based kinematics configuration

- **Wheels** (`wheels.proto`): Wheel-based kinematics
  - `DifferentialConfig` - Differential drive
  - `MecanumConfig` - Mecanum drive
  - `Steer4Config` - 4-wheel steering
  - `Steer4DualConfig` - 4-wheel dual steering
  - `Steer6Config` - 6-wheel articulated steering
  - `WheelsConfig` - Union type for wheel configurations

- **RigidBody** (`rigidbody.proto`): Rigid body kinematics
  - `RigidBodyConfig` - Mass, inertia tensor, gains

- **Types** (`types.proto`): Common kinematics types
  - `Config` - Model configuration
  - `Dimensions` - State/control dimensionality
  - `Capabilities` - Kinematic properties
  - `KinematicsConfig` - Union type for all kinematics

- **State** (`state.proto`): Kinematics state and sensors
  - `State`, `Target`, `SensorData`, `IMUData`, `BodyPose`

### Filter Types (`types/filter/`)

- **SLAM** (`slam.proto`): SLAM data structures
  - `OccupancyGrid` - Occupancy grid map
  - `SLAMPose` - Robot pose (position + orientation)
  - `SLAMConfig` - SLAM filter configuration
  - `SLAMMeasurement` - Distance measurements from ray-based sensors
  - `SLAMState` - SLAM filter state with covariance

### Interaction Types (`types/interaction/`)

- **Events** (`events.proto`): Mouse, keyboard, touch, and joystick input events
  - `MouseEvent` - Mouse events (press, release, move, scroll, enter, leave)
  - `MouseButton` - Mouse button enum (left, right, middle, extra)
  - `MouseEventType` - Mouse event type enum
  - `KeyboardEvent` - Keyboard events (press, release, type) with key as string (e.g., "a", "Enter", "F1", "ArrowUp")
  - `KeyboardEventType` - Keyboard event type enum
  - `TouchEvent` - Multi-touch screen events (down, move, up, cancel)
  - `TouchPoint` - Single touch point with position (Vector2D), pressure, size, orientation, velocity (Vector2D), acceleration (Vector2D), displacement (Vector2D)
  - `TouchEventType` - Touch event type enum
  - `ControllerEvent` - Controller events (value changed, boolean changed, trim changed, connect, disconnect)
  - `ControllerState` - Complete controller state with multiple controls and trims
  - `Control` - Individual control with axis array, buttons array, velocities array, accelerations array
  - `Trim` - Trim control (e.g., trim wheel, dial) with value and velocity
  - `ControlEventType` - Controller event type enum

- **Messages** (`messages.proto`): User messages and commands in text form
  - `UserMessage` - Text messages, commands, queries, notifications
  - `MessageType` - Message type enum

- **Requests** (`requests.proto`): User requests and responses with arbitrary data
  - `UserRequest` - User request with `google.protobuf.Any` payload
  - `UserResponse` - Response with text, timestamp, and `google.protobuf.Any` payload
  - `RequestType` - Request type enum (action, query, config, data)
  - `ResponseStatus` - Response status enum (success, error, partial, pending)

### Core Types (`types/core/`)

- **Frame**: Multiframe frame with:
  - `index` (int64) - Frame index in sequence
  - `timestamp` (int64) - Timestamp in nanoseconds
  - `metadata` (map[string]string) - Additional metadata
  - `tensors` (repeated Tensor) - Tensor data

- **ArbitraryValue**: Encapsulates arbitrary types for local data transfer:
  - `type_name` - Fully qualified type name
  - `kind` - Kind discriminator
  - `reference` - Reference/pointer value (platform-specific)
  - `metadata` - Additional metadata

  Used for zero-copy, no-marshalling scenarios where data is passed within the same process or via shared memory.

## Usage

### Generating Code

Use `buf generate` to generate code from the proto definitions:

```bash
cd proto
buf generate
```

### Importing in Other Proto Files

```protobuf
import "types/math/tensor.proto";
import "types/core/frame.proto";

message MyMessage {
  types.math.Tensor data = 1;
  types.core.Frame frame = 2;
}
```

## Migration Notes

The proto structure has been reorganized to match the `x/math/control` folder structure:

### Old Structure (Removed)
- `types/kinematics/config.proto` - Consolidated kinematics configs
- `types/kinematics/state.proto` - Consolidated kinematics state

### New Structure
- `types/control/kinematics/` - Organized by implementation type:
  - `joints.proto` - Joint-based kinematics
  - `wheels.proto` - Wheel-based kinematics
  - `rigidbody.proto` - Rigid body kinematics
  - `types.proto` - Common types
  - `state.proto` - State and sensors

### Import Updates
If you were using the old files, update your imports:
- `types.kinematics.*` → `types.control.kinematics.*`
- Old kinematics configs are now in specific files (joints, wheels, rigidbody)
- Path and motion types are now in `types.control.*`
- SLAM types are in `types.filter.*`

## Design Principles

1. **Logical Organization**: Types are grouped by domain (math, neural networks, kinematics, core)
2. **Clear Naming**: Package names match directory structure
3. **Comprehensive Coverage**: All major types from `x/math` are represented
4. **Zero-Copy Support**: `ArbitraryValue` enables local transfer without marshalling
5. **Extensibility**: Union types (oneof) allow adding new configurations without breaking changes

