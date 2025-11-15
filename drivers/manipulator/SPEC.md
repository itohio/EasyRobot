# Manipulator Driver Specification

## Purpose

The manipulator driver is designed to test manipulator **backwards (inverse) kinematics** on embedded hardware. It provides three build variants:

1. **Raw Mode**: Direct joint angle control (no kinematics)
2. **Planar Mode**: Planar kinematics - receives end effector position, calculates joint angles using planar inverse kinematics
3. **DH Mode**: Denavit-Hartenberg kinematics - receives end effector position, calculates joint angles using DH inverse kinematics

## Build Variants

### Raw Mode (Default)
- **Build tags**: `sam,xiao` (no additional tags)
- **Intent path**: `easyrobot.manipulator.raw`
- **Control**: Direct joint angle targets (degrees)
- **Use case**: Testing servo motors without kinematics calculations

### Planar Mode
- **Build tags**: `sam,xiao,planar`
- **Intent path**: `easyrobot.manipulator.planar`
- **Control**: End effector position target (3D position, optional orientation)
- **Kinematics**: Planar inverse kinematics calculates joint angles
- **Use case**: Testing planar manipulators where all joints lie in a single plane

### DH Mode
- **Build tags**: `sam,xiao,dh`
- **Intent path**: `easyrobot.manipulator.dh`
- **Control**: End effector position target (3D position with orientation)
- **Kinematics**: Denavit-Hartenberg inverse kinematics calculates joint angles
- **Use case**: Testing general 3D manipulators using DH parameters

## Build Commands

```bash
# Raw mode (default)
tinygo flash -target=xiao -tags logless ./drivers/manipulator

# Planar mode
tinygo flash -target=xiao -tags logless,planar ./drivers/manipulator

# DH mode
tinygo flash -target=xiao -tags logless,dh ./drivers/manipulator
```

## Architecture

The manipulator system consists of two programs working in tandem:

### Driver Program (`drivers/manipulator/`)
- Runs on XIAO embedded board
- Controls servo motors
- Implements inverse kinematics (planar/dh modes)
- Publishes state via DNDM Intent
- Receives config and target commands via DNDM Interest

### Client Program (`cmd/manipulator/`)
- Runs on host computer
- Connects to XIAO board via serial
- Subscribes to state via DNDM Interest
- Sends config and target commands via DNDM Intent
- Provides interactive command interface

## DNDM Communication

### Intent/Interest Pattern

**Driver (XIAO Board)**:
- **Intent** (Publisher): `easyrobot.manipulator.{mode}` → Publishes `ManipulatorState`
- **Interest** (Subscriber): `easyrobot.manipulator.{mode}.config` → Receives `ManipulatorConfig`
- **Interest** (Subscriber): `easyrobot.manipulator.{mode}.target` → Receives `ManipulatorTarget`

**Client (Host Computer)**:
- **Interest** (Subscriber): `easyrobot.manipulator.{mode}` → Receives `ManipulatorState`
- **Intent** (Publisher): `easyrobot.manipulator.{mode}.config` → Sends `ManipulatorConfig`
- **Intent** (Publisher): `easyrobot.manipulator.{mode}.target` → Sends `ManipulatorTarget`

### Intent Paths
The driver declares intent based on build mode:
- Raw: `easyrobot.manipulator.raw`
- Planar: `easyrobot.manipulator.planar`
- DH: `easyrobot.manipulator.dh`

### Topics

#### State Publisher
All modes publish state on: `easyrobot.manipulator.{mode}`

**Message**: `types.control.ManipulatorState`
- `joint_angles`: Current joint angles (degrees)
- `joint_velocities`: Current joint velocities (degrees/sec)
- `timestamp`: Timestamp in nanoseconds

#### Config Consumer
All modes subscribe to: `easyrobot.manipulator.{mode}.config`

**Message**: `types.control.ManipulatorConfig`
- `motors`: Motor configurations (always required)
- `joints`: Joint kinematics configuration (required for planar/dh modes, ignored in raw mode)
- `motion`: Motion constraints per joint (velocity, acceleration, jerk limits)

#### Target Consumer
All modes subscribe to: `easyrobot.manipulator.{mode}.target`

**Message**: `types.control.ManipulatorTarget`

**For Raw Mode**:
- Uses `joint_angles` field (degrees)

**For Planar/DH Modes**:
- Uses `end_effector_position` field (3D position in meters)
- Uses `end_effector_orientation` field (quaternion, optional for planar, required for DH)

## Control Flow

### Raw Mode
1. Receive `ManipulatorConfig` → Configure motors
2. Receive `ManipulatorTarget` with `joint_angles` → Set motion targets directly
3. Motion controller updates servo positions
4. Publish `ManipulatorState` with current joint angles

### Planar/DH Modes
1. Receive `ManipulatorConfig` with `joints` → Configure motors and initialize kinematics
2. Receive `ManipulatorTarget` with `end_effector_position` → Calculate joint angles using inverse kinematics
3. Set calculated joint angles as motion targets
4. Motion controller updates servo positions
5. Publish `ManipulatorState` with current joint angles

## Hardware

### Supported Hardware
- **Platform**: Seeed Studio XIAO series (SAM D21)
- **Interface**: UART for DNDM communication
- **Motors**: 3 servo motors on pins D8, D9, D10
- **Timer Mapping**: See `xiao.go` for PWM timer assignments

### Pin Configuration
```go
Motor 0: Pin D8  → TCC1
Motor 1: Pin D9  → TCC0
Motor 2: Pin D10 → TCC1
```

## Motion Control

All modes use VAJ (Velocity-Acceleration-Jerk) motion profiles for smooth motion:
- Motion profiles are created per joint
- Default limits: 100 deg/s velocity, 100 deg/s² acceleration, 100 deg/s³ jerk
- Update rate: 100ms (10 Hz)

## Kinematics Integration

### Planar Kinematics
- **Package**: `x/math/control/kinematics/joints/planar`
- **Configuration**: `types.control.kinematics.JointsConfig` with `planar_joints`
- **Inverse Kinematics**: Calculates joint angles from end effector position in XZ plane
- **Forward Kinematics**: (Optional) Validates IK solutions

### DH Kinematics
- **Package**: `x/math/control/kinematics/joints/dh`
- **Configuration**: `types.control.kinematics.JointsConfig` with `dh_params`
- **Inverse Kinematics**: Calculates joint angles from end effector pose (position + orientation)
- **Forward Kinematics**: (Optional) Validates IK solutions

## Proto Definitions

See `proto/types/control/manipulator.proto`:
- `MotorConfig`: Single servo motor configuration
- `ManipulatorConfig`: Full manipulator configuration
- `ManipulatorTarget`: Target (joint angles or end effector position)
- `ManipulatorState`: Current state

## Client Usage

### Build and Run Client

```bash
cd cmd/manipulator
go build
./manipulator --help
```

### Client Flags

- `--port <path>`: Serial port path (default: `/dev/ttyACM0`)
- `--baud <rate>`: Serial baud rate (default: `115200`)
- `--mode <mode>`: Build mode: `raw`, `planar`, or `dh` (default: `raw`)
- `--list`: List available serial ports
- `--help`: Show help message

### Client Commands

**Interactive Mode**:
- `config`: Send manipulator configuration (motors, kinematics, motion constraints)
- `target <joint1> [joint2] [joint3]`: Set joint angle targets in degrees (raw mode only)
- `pose <x> <y> <z> [w x y z]`: Set end effector pose (planar/dh modes only)
  - Position: x, y, z in meters
  - Orientation: quaternion [w, x, y, z] (optional for planar, required for dh)
- `state`: Show last received state from driver
- `quit`: Exit client

### Example Usage

```bash
# Raw mode - direct joint control
./manipulator --mode raw --port /dev/ttyACM0
> config
> target 45 0 -45

# Planar mode - end effector position control
./manipulator --mode planar --port /dev/ttyACM0
> config
> pose 0.1 0.0 0.15

# DH mode - end effector pose control (with orientation)
./manipulator --mode dh --port /dev/ttyACM0
> config
> pose 0.1 0.0 0.15 1.0 0.0 0.0 0.0
```

## Testing Inverse Kinematics

The primary purpose is to test backwards (inverse) kinematics:

1. **Planar Mode**:
   - Client sends end effector position target via `pose` command
   - Driver calculates joint angles using planar IK
   - Joints move to calculated positions
   - Driver publishes state, client displays it
   - Verify end effector reaches target position

2. **DH Mode**:
   - Client sends end effector pose target (position + orientation) via `pose` command
   - Driver calculates joint angles using DH IK
   - Joints move to calculated positions
   - Driver publishes state, client displays it
   - Verify end effector reaches target pose

## Dependencies

### Required (TinyGo Compatible)
- `github.com/itohio/dndm` - DNDM communication framework
- `tinygo.org/x/drivers` - TinyGo hardware drivers

### Required (To Be Migrated)
- `github.com/itohio/EasyRobot/x/devices/servos` - Servo motor control (currently using `pkg/control/actuator/servos`)
- `github.com/itohio/EasyRobot/x/math/control/kinematics/joints/planar` - Planar kinematics (integration pending)
- `github.com/itohio/EasyRobot/x/math/control/kinematics/joints/dh` - DH kinematics (integration pending)

### Already Using
- `github.com/itohio/EasyRobot/x/math/control/motion` - VAJ motion profiles
- `github.com/itohio/EasyRobot/types/control` - Proto types

## TODOs

1. Migrate servo controls from `pkg/control/actuator/servos` to `x/devices/servos`
2. Integrate planar kinematics from `x/math/control/kinematics/joints/planar`
3. Integrate DH kinematics from `x/math/control/kinematics/joints/dh`
4. Add forward kinematics validation (optional)
5. Handle workspace limits and unreachable targets
6. Add error reporting for IK failures

## Design Principles

- **Build-time selection**: Kinematics mode selected at compile time via build tags (driver) or flags (client)
- **Separation of concerns**: Motion control (VAJ) separate from kinematics
- **Proto-driven communication**: All messages use proto definitions
- **DNDM-based**: Uses DNDM for distributed communication via Intent/Interest pattern
- **Intent-based routing**: Different intent paths for different modes
- **Client-server architecture**: Client on host, driver on embedded device, communicate via serial DNDM
- **Interactive testing**: Client provides interactive commands for testing inverse kinematics

