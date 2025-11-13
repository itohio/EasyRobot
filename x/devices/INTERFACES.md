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

