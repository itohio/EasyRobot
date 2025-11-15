# XIAO Device Package

This package provides XIAO-specific device implementations for the Seeed Studio XIAO board (SAM D21 microcontroller).

## PWM Device

The XIAO PWM device provides PWM (Pulse Width Modulation) functionality using the SAM D21's TCC (Timer/Counter Controller) peripherals.

### Features

- **Multiple PWM Channels**: Supports PWM on multiple pins using TCC0, TCC1, and TCC2 peripherals
- **Frequency Control**: Configurable PWM frequency (default: 50Hz for servos)
- **Duty Cycle Control**: Set duty cycle as percentage (0.0 to 1.0)
- **Servo Support**: Direct pulse width control in microseconds (500-2500 µs for servos)
- **Pin Mapping**: Automatic mapping of XIAO pins to TCC channels

### Supported PWM Pins

**Verified Pins** (from manipulator driver):
- `D8`: TCC1 Channel 0
- `D9`: TCC0 Channel 0
- `D10`: TCC1 Channel 1

**Additional Pins** (verify against XIAO pinout):
- `D2-D7`: Various TCC0/TCC1 channels
- `A0-A2`: TCC1 channels
- `D0-D1`: Potentially TCC2 channels (unverified)

See `xiao_pwm.go` for complete pin mapping.

### Usage

```go
import (
    "machine"
    "github.com/itohio/EasyRobot/x/devices"
    xiao "github.com/itohio/EasyRobot/x/devices/xiao"
)

// Create PWM device
pwm := xiao.NewPWMDevice()

// Configure for 50Hz (servo frequency)
pwm.Configure(50)

// Get PWM channel for a pin
pin := devices.NewTinyGoPin(machine.D8)
channel, err := pwm.Channel(pin)
if err != nil {
    // Handle error
}

// Set duty cycle (0.0 to 1.0)
channel.Set(0.5) // 50% duty cycle

// Or set pulse width for servos (500-2500 microseconds)
channel.SetMicroseconds(1500) // 1.5ms = center position

// Stop PWM
channel.Stop()
```

### Implementation Details

- Uses SAM D21 TCC peripherals (TCC0, TCC1, TCC2)
- Automatic TCC configuration and management
- Shared TCC peripherals: Multiple pins can share the same TCC (different channels)
- Thread-safe: Uses mutex to protect channel management
- Frequency shared: All channels on the same TCC share the same frequency

### Build Tags

- `sam && xiao`: XIAO-specific implementation
- `!sam || !xiao`: Stub implementation (returns errors)

