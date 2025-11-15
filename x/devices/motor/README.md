# Motor Device Package

This package provides motor control with **PID speed control**, encoder feedback, PWM control, and direction control.

## Overview

The motor package provides:
- **PID Speed Control**: Automatic speed regulation using encoder feedback
- **Multiple Motor Types**: Supports different motor driver configurations
- **Encoder Integration**: Uses encoder for speed feedback
- **PWM Control**: Controls motor speed via PWM
- **Direction Control**: Manages motor direction via pins

## Motor Types

The package supports three motor driver configurations:

### 1. TypeDirPWM: One Direction Pin + One PWM Pin

- **Dir Pin**: Controls direction (high = forward, low = reverse)
- **PWM Pin**: Controls speed (0-100% duty cycle)

```
Dir Pin ──┐
          ├── Motor Driver ── Motor
PWM Pin ──┘
```

### 2. TypeABPWM: Two Pins (A and B) Both with PWM

- **PinA**: PWM for forward direction (PinB = 0 when forward)
- **PinB**: PWM for reverse direction (PinA = 0 when reverse)
- **Speed and Direction**: Controlled by relative PWM duty cycles

```
PinA (PWM) ──┐
             ├── Motor Driver ── Motor
PinB (PWM) ──┘
```

### 3. TypeABDirPWM: Two Direction Pins + One PWM Pin

- **PinA**: Direction pin (high = forward, low = reverse)
- **PinB**: Direction pin (opposite of PinA)
- **PWM Pin**: Controls speed (0-100% duty cycle)

```
PinA (Dir) ──┐
PinB (Dir) ──┤
             ├── Motor Driver ── Motor
PWM Pin ─────┘
```

## Usage

### Basic Usage

```go
import (
    "machine"
    "github.com/itohio/EasyRobot/x/devices"
    "github.com/itohio/EasyRobot/x/devices/encoder"
    "github.com/itohio/EasyRobot/x/devices/motor"
    xiao "github.com/itohio/EasyRobot/x/devices/xiao"
)

// Create PWM device
pwm := xiao.NewPWMDevice()

// Create encoder
encA := devices.NewTinyGoPin(machine.D2)
encB := devices.NewTinyGoPin(machine.D3)
enc := encoder.New(encA, encB, encoder.DefaultConfig())
enc.Configure()

// Configure motor
config := motor.DefaultConfig()
config.Type = motor.TypeDirPWM
config.Dir = devices.NewTinyGoPin(machine.D4)
config.PWM = devices.NewTinyGoPin(machine.D5)
config.Encoder = enc
config.PIDGains.P = 1.0
config.PIDGains.I = 0.1
config.PIDGains.D = 0.01
config.MaxRPM = 100

// Create motor
mot, err := motor.New(pwm, config)
if err != nil {
    // Handle error
}

// Enable motor control
mot.Enable()

// Set target speed (RPM)
mot.SetSpeed(50) // Forward at 50 RPM

// Later...
mot.SetSpeed(-30) // Reverse at 30 RPM

// Stop
mot.Disable()
```

### MotorArray Example

```go
// Create motor array with multiple motors
configs := []motor.Config{
    {
        Type:        motor.TypeDirPWM,
        Dir:         devices.NewTinyGoPin(machine.D4),
        PWM:         devices.NewTinyGoPin(machine.D5),
        Encoder:     enc1,
        PIDGains:    struct{ P, I, D float32 }{P: 1.0, I: 0.1, D: 0.01},
        MaxRPM:      100,
    },
    {
        Type:        motor.TypeDirPWM,
        Dir:         devices.NewTinyGoPin(machine.D6),
        PWM:         devices.NewTinyGoPin(machine.D7),
        Encoder:     enc2,
        PIDGains:    struct{ P, I, D float32 }{P: 1.0, I: 0.1, D: 0.01},
        MaxRPM:      100,
    },
}

// Create motor array
array, err := motor.NewMotorArray(pwm, configs)
if err != nil {
    // Handle error
}

// Enable all motors
array.Enable()

// Set speeds for all motors (RPM)
speeds := []float32{50, -30} // Forward at 50 RPM, reverse at 30 RPM
array.SetSpeeds(speeds)

// Get current speeds
currentSpeeds := array.Speeds()

// Disable all motors
array.Disable()
```

### TypeDirPWM Example

```go
config := motor.Config{
    Type:        motor.TypeDirPWM,
    Dir:         devices.NewTinyGoPin(machine.D4),
    PWM:         devices.NewTinyGoPin(machine.D5),
    Encoder:     enc,
    PIDGains:    struct{ P, I, D float32 }{P: 1.0, I: 0.1, D: 0.01},
    MaxOutput:   1.0,
    SamplePeriod: 0.01,
    MaxRPM:      100,
}
```

### TypeABPWM Example

```go
config := motor.Config{
    Type:        motor.TypeABPWM,
    PinA:        devices.NewTinyGoPin(machine.D4),
    PinB:        devices.NewTinyGoPin(machine.D5),
    Encoder:     enc,
    PIDGains:    struct{ P, I, D float32 }{P: 1.0, I: 0.1, D: 0.01},
    MaxOutput:   1.0,
    SamplePeriod: 0.01,
    MaxRPM:      100,
}
```

### TypeABDirPWM Example

```go
config := motor.Config{
    Type:        motor.TypeABDirPWM,
    PinA:        devices.NewTinyGoPin(machine.D4),
    PinB:        devices.NewTinyGoPin(machine.D5),
    PWM:         devices.NewTinyGoPin(machine.D6),
    Encoder:     enc,
    PIDGains:    struct{ P, I, D float32 }{P: 1.0, I: 0.1, D: 0.01},
    MaxOutput:   1.0,
    SamplePeriod: 0.01,
    MaxRPM:      100,
}
```

## PID Control

The motor uses a PID controller to maintain the target speed:

- **Input**: Current speed from encoder (RPM)
- **Target**: Desired speed (RPM)
- **Output**: PWM duty cycle (normalized to [-1, 1])

The PID controller runs at the specified `SamplePeriod` (default: 10ms).

### Tuning PID Gains

- **P (Proportional)**: Direct response to error (higher = faster response, but can oscillate)
- **I (Integral)**: Eliminates steady-state error (higher = faster settling, but can overshoot)
- **D (Derivative)**: Reduces overshoot (higher = more damping, but can be noisy)

Typical starting values:
- P: 1.0
- I: 0.1
- D: 0.01

## Architecture

### Motor

The `Motor` struct provides:
- **New()**: Creates a motor with configuration
- **SetSpeed()**: Sets target speed in RPM (positive = forward, negative = reverse)
- **Speed()**: Returns current speed from encoder
- **TargetSpeed()**: Returns target speed in RPM
- **Enable()**: Starts PID control loop
- **Disable()**: Stops motor and control loop
- **Close()**: Cleans up resources

### MotorArray

The `MotorArray` struct controls an **array of motors**:
- **NewMotorArray()**: Creates an array for multiple motors
- **Enable()**: Enables all motors in the array
- **Disable()**: Disables all motors in the array
- **SetSpeeds()**: Sets target speeds for all motors in the array (RPM)
- **Speeds()**: Returns current speeds for all motors in the array
- **TargetSpeeds()**: Returns target speeds for all motors in the array
- **Close()**: Cleans up resources

**Key Features**:
- Controls multiple motors simultaneously
- Each motor runs its own PID control loop
- Thread-safe operations with mutex protection
- Convenient batch operations for arrays of motors

### Control Loop

The motor runs a control loop that:
1. Reads current speed from encoder
2. Updates PID controller
3. Converts PID output to PWM duty cycle
4. Sets direction pins (if needed)
5. Applies PWM to motor

## Dependencies

- `github.com/itohio/EasyRobot/x/devices` - Device interfaces (PWM, Pin)
- `github.com/itohio/EasyRobot/x/devices/encoder` - Encoder for feedback
- `github.com/itohio/EasyRobot/x/math/control/pid` - PID controller
- Platform-specific PWM devices (e.g., `x/devices/xiao` for XIAO board)

## Thread Safety

All motor operations are thread-safe using mutex locks.

