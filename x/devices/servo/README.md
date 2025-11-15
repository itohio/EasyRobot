# Servo Package

This package provides servo motor control using PWM (Pulse Width Modulation) devices.

## Overview

The servo package provides:
- **Motor Configuration**: Options pattern for configuring individual servo motors
- **ServoArray**: Multi-channel servo array that controls multiple servo motors simultaneously
- **PWM-Based Control**: Uses PWM devices to control servo pulse widths

**Note**: This package is designed for controlling **arrays of servos** via the `ServoArray` type. All operations work on arrays of servo motors.

## Usage

### Basic Usage

```go
import (
    "machine"
    "github.com/itohio/EasyRobot/x/devices"
    "github.com/itohio/EasyRobot/x/devices/servo"
    xiao "github.com/itohio/EasyRobot/x/devices/xiao"
)

// Create PWM device
pwm := xiao.NewPWMDevice()

// Configure motors
motors := []servo.Motor{
    servo.NewMotorConfig(
        servo.WithPin(uint32(machine.D8)),
        servo.WithMicroseconds(500, 2500, 1500, 180),
    ),
    servo.NewMotorConfig(
        servo.WithPin(uint32(machine.D9)),
        servo.WithMicroseconds(500, 2500, 1500, 180),
    ),
}

// Create servo array for array of servos
array, err := servo.NewServoArray(pwm, motors)
if err != nil {
    // Handle error
}

// Set servo positions (angles in degrees)
angles := []float32{90, 45}
if err := array.Set(angles); err != nil {
    // Handle error
}

// Stop servos (move to default positions)
if err := array.Stop(); err != nil {
    // Handle error
}
```

### Motor Configuration Options

```go
// Basic configuration with just pin
motor := servos.NewMotorConfig(
    servos.WithPin(uint32(machine.D8)),
)

// Full configuration
motor := servos.NewMotorConfig(
    servos.WithPin(uint32(machine.D8)),
    servos.WithMicroseconds(
        500,   // min_us: minimum pulse width in microseconds
        2500,  // max_us: maximum pulse width in microseconds
        1500,  // default_us: default/center pulse width
        180,   // max_angle: maximum angle in degrees
    ),
)
```

### Angle to Microseconds Conversion

```go
motor := servos.NewMotorConfig(
    servos.WithPin(uint32(machine.D8)),
    servos.WithMicroseconds(500, 2500, 1500, 180),
)

// Convert angle to microseconds
us := motor.AngleToMicroseconds(90) // Returns microseconds for 90 degrees

// Convert microseconds to angle
angle := motor.MicrosecondsToAngle(1500) // Returns angle for 1500 microseconds
```

## Architecture

### Motor

The `Motor` struct represents a single servo motor configuration:
- **Pin**: GPIO pin number
- **MinUs/MaxUs/DefaultUs**: Pulse width limits in microseconds
- **MaxAngle/Default**: Angle limits in degrees
- **AngleToMicroseconds()**: Convert angle to pulse width
- **MicrosecondsToAngle()**: Convert pulse width to angle

### ServoArray

The `ServoArray` struct controls an **array of servo motors**:
- **NewServoArray()**: Creates an array for multiple servos
- **Configure()**: Configure the array of motors (count must match)
- **Set()**: Set positions for all servos in the array (angles in degrees)
- **Stop()**: Stop all servos in the array (move to default positions)

**Key Features**:
- Controls multiple servo motors simultaneously
- Uses `devices.PWMDevice` for PWM control
- Thread-safe operations with mutex protection
- Automatic PWM channel management for each servo
- All operations work on arrays (not individual servos)

## Array of Servos

This package is specifically designed for controlling **arrays of servo motors**. All operations work on the entire array:

- **NewActuator()**: Takes an array of motor configurations
- **Set()**: Takes an array of angles (one per motor)
- **Configure()**: Updates the entire array (count must match)
- **Stop()**: Stops all servos in the array

For single servo control, use an array of length 1.

## Dependencies

- `github.com/itohio/EasyRobot/x/devices` - Device interfaces (PWM, Pin)
- Platform-specific PWM devices (e.g., `x/devices/xiao` for XIAO board)

## Build Tags

- `tinygo`: TinyGo-specific implementation
- `!tinygo`: Stub implementation (returns errors)

