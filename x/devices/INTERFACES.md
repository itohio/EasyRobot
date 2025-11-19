# Device Interfaces

This package provides platform-agnostic interfaces for hardware communication that work with both TinyGo and standard Go (Linux/Raspberry Pi).

## Interfaces

All interfaces are defined in `x/devices` directly (not in subpackages), following the pattern from `tinygo.org/x/drivers`.

### I2C Interface

```go
type I2C interface {
    ReadRegister(addr uint8, r uint8, buf []byte) error
    WriteRegister(addr uint8, r uint8, buf []byte) error
    Tx(addr uint16, w, r []byte) error
}
```

**Implementations:**
- `TinyGoI2C` - Wraps `machine.I2C` (TinyGo)
- `LinuxI2C` - Uses Linux `/dev/i2c-*` (Raspberry Pi)
- `StubI2C` - Stub for unsupported platforms

**Usage:**
```go
// TinyGo
machineI2C := machine.I2C0
machineI2C.Configure(machine.I2CConfig{})
bus := devices.NewTinyGoI2C(machineI2C)

// Linux (Raspberry Pi)
bus, err := devices.NewLinuxI2C("/dev/i2c-1")
```

### SPI Interface

```go
type SPI interface {
    Tx(w, r []byte) error
    Transfer(b byte) (byte, error)
}
```

**Implementations:**
- `TinyGoSPI` - Wraps `machine.SPI` (TinyGo)
- `LinuxSPI` - Uses Linux `/dev/spidev*` (Raspberry Pi)
- `StubSPI` - Stub for unsupported platforms

**Usage:**
```go
// TinyGo
machineSPI := machine.SPI0
machineSPI.Configure(machine.SPIConfig{})
bus := devices.NewTinyGoSPI(machineSPI)

// Linux (Raspberry Pi)
bus, err := devices.NewLinuxSPI("/dev/spidev0.0")
```

### Serial (UART) Interface

```go
type Serial interface {
    io.Reader
    io.Writer
    Buffered() int
}
```

**Implementations:**
- `TinyGoSerial` - Wraps `machine.UART` (TinyGo)
- `LinuxSerial` - Uses Linux serial ports (Raspberry Pi)
- `StubSerial` - Stub for unsupported platforms

**Usage:**
```go
// TinyGo
machineUART := machine.UART0
machineUART.Configure(machine.UARTConfig{})
serial := devices.NewTinyGoSerial(machineUART)

// Linux (Raspberry Pi)
serial, err := devices.NewLinuxSerial("/dev/ttyAMA0")
```

### Pin Interface

```go
type Pin interface {
    Get() bool
    Set(value bool)
}
```

**Implementations:**
- `TinyGoPin` - Wraps `machine.Pin` (TinyGo)
- `LinuxPin` - Uses Linux sysfs GPIO (Raspberry Pi)
- `StubPin` - Stub for unsupported platforms

**Usage:**
```go
// TinyGo
machinePin := machine.D2
machinePin.Configure(machine.PinConfig{Mode: machine.PinOutput})
pin := devices.NewTinyGoPin(machinePin)

// Linux (Raspberry Pi)
// Note: GPIO must be exported first: echo 18 > /sys/class/gpio/export
pin, err := devices.NewLinuxPin(18)
```

### PWM Interface

```go
type PWM interface {
    Set(duty float32) error          // duty cycle 0.0 to 1.0
    SetMicroseconds(us uint32) error // pulse width in microseconds
    Stop() error                      // stop PWM output
}

type PWMDevice interface {
    Channel(pin Pin) (PWM, error)
    Configure(frequency uint32) error
    SetFrequency(frequency uint32) error
}
```

**Implementations:**
- `xiao.NewPWMDevice()` - XIAO board using TCC peripherals (TinyGo, SAM D21)
- Other platform-specific implementations can be added

**Usage:**
```go
// XIAO board
import "github.com/itohio/EasyRobot/x/devices/xiao"

pwm := xiao.NewPWMDevice()
pwm.Configure(50) // 50Hz for servos

pin := devices.NewTinyGoPin(machine.D8)
channel, err := pwm.Channel(pin)
if err != nil {
    // Handle error
}

// Set duty cycle (0.0 to 1.0)
channel.Set(0.5) // 50% duty cycle

// Or set pulse width for servos (500-2500 microseconds)
channel.SetMicroseconds(1500) // 1.5ms = center position
```

## Device Driver Usage

Device drivers should accept these interfaces instead of `machine.*` types directly:

```go
// Device driver example
type Device struct {
    bus devices.I2C
    address uint8
}

func New(bus devices.I2C, address uint8) *Device {
    return &Device{bus: bus, address: address}
}
```

This allows the same device driver to work with:
- TinyGo microcontrollers (using `NewTinyGoI2C(machine.I2C0)`)
- Raspberry Pi Linux (using `NewLinuxI2C("/dev/i2c-1")`)
- Other platforms (with appropriate implementations)

## Build Tags

The implementations use build tags to select the correct code:

- `tinygo` - TinyGo implementations
- `!tinygo && linux` - Linux implementations
- `!tinygo && !linux` - Stub implementations

## Compatibility

These interfaces are compatible with `tinygo.org/x/drivers` patterns:
- `devices.I2C` matches `drivers.I2C` interface
- `devices.SPI` matches `drivers.SPI` interface
- `devices.Serial` matches `drivers.UART` interface

This means device drivers can work with both `tinygo.org/x/drivers` and this package's implementations.

### Spectrometer Interface

```go
type Spectrometer interface {
    NumWavelengths() int
    Wavelengths(dst vecTypes.Vector) vecTypes.Vector
    Measure(ctx context.Context, dst matTypes.Matrix) error
    WaitMeasurement(ctx context.Context, dst matTypes.Matrix) error
}
```

**Implementations:**
- `cr30.Device` - CR30 colorimeter/spectrometer (via serial port)

**Usage:**
```go
import (
    "github.com/itohio/EasyRobot/x/devices"
    "github.com/itohio/EasyRobot/x/devices/cr30"
    "github.com/itohio/EasyRobot/x/math/mat"
    "github.com/itohio/EasyRobot/x/math/vec"
)

// Create serial connection
serial, err := devices.NewSerialWithConfig("/dev/ttyUSB0", config)
if err != nil {
    // Handle error
}

// Create spectrometer device
dev := cr30.New(serial)
if err := dev.Connect(); err != nil {
    // Handle error
}

// Get number of wavelengths
numWl := dev.NumWavelengths()

// Get wavelengths
wlVec := vec.New(numWl)
dev.Wavelengths(wlVec)

// Measure with 1-row matrix (SPD values only)
dst := mat.New(1, numWl)
err := dev.Measure(ctx, dst)
spd := dst.Row(0).(vec.Vector)

// Measure with 2-row matrix (wavelengths + SPD)
dst2 := mat.New(2, numWl)
err = dev.Measure(ctx, dst2)
wavelengths := dst2.Row(0).(vec.Vector)
spd := dst2.Row(1).(vec.Vector)

// Wait for user-initiated measurement
err = dev.WaitMeasurement(ctx, dst)
```

