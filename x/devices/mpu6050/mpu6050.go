// Package mpu6050 provides a driver for the MPU6050 6-axis accelerometer and gyroscope.
//
// Datasheet: https://www.invensense.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf
package mpu6050

import (
	"github.com/itohio/EasyRobot/x/devices"
)

// DefaultAddress is the default I2C address for the MPU6050
const DefaultAddress = 0x68

// Register addresses
const (
	SelfTestX    = 0x0D
	SelfTestY    = 0x0E
	SelfTestZ    = 0x0F
	SelfTestA    = 0x10
	SMPLRTDiv    = 0x19
	Config       = 0x1A
	GyroConfig   = 0x1B
	AccelConfig  = 0x1C
	FIFOEnable   = 0x23
	IntPinConfig = 0x37
	IntEnable    = 0x38
	IntStatus    = 0x3A
	AccelXOutH   = 0x3B
	AccelXOutL   = 0x3C
	AccelYOutH   = 0x3D
	AccelYOutL   = 0x3E
	AccelZOutH   = 0x3F
	AccelZOutL   = 0x40
	TempOutH     = 0x41
	TempOutL     = 0x42
	GyroXOutH    = 0x43
	GyroXOutL    = 0x44
	GyroYOutH    = 0x45
	GyroYOutL    = 0x46
	GyroZOutH    = 0x47
	GyroZOutL    = 0x48
	UserCtrl     = 0x6A
	PWRMgmt1     = 0x6B
	PWRMgmt2     = 0x6C
	FIFOCountH   = 0x72
	FIFOCountL   = 0x73
	FIFORW       = 0x74
	WhoAmI       = 0x75
)

// WhoAmIValue is the expected value from the WhoAmI register
const WhoAmIValue = 0x68

// AccelerometerData contains accelerometer readings.
type AccelerometerData struct {
	X, Y, Z int16
}

// GyroscopeData contains gyroscope readings.
type GyroscopeData struct {
	X, Y, Z int16
}

// Device wraps an I2C connection to an MPU6050 device.
type Device struct {
	bus     devices.I2C
	address uint8
}

// New creates a new MPU6050 connection. The I2C bus must already be configured.
// The bus can be either a TinyGo machine.I2C (wrapped with devices.NewI2C)
// or a Linux I2C bus (created with devices.NewI2C).
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{
		bus:     bus,
		address: address,
	}
}

// Configure initializes the device.
func (d *Device) Configure() error {
	// Wake up the device
	if err := d.write8(PWRMgmt1, 0x00); err != nil {
		return err
	}

	// Set sample rate divider (1kHz / (1 + 7) = 125Hz)
	if err := d.write8(SMPLRTDiv, 7); err != nil {
		return err
	}

	// Configure accelerometer (±2g)
	if err := d.write8(AccelConfig, 0x00); err != nil {
		return err
	}

	// Configure gyroscope (±250°/s)
	if err := d.write8(GyroConfig, 0x00); err != nil {
		return err
	}

	// Configure DLPF (Digital Low Pass Filter)
	if err := d.write8(Config, 0x06); err != nil {
		return err
	}

	return nil
}

// ReadAccelerometer reads the accelerometer values.
func (d *Device) ReadAccelerometer() (*AccelerometerData, error) {
	data := make([]byte, 6)
	if err := d.bus.Tx(uint16(d.address), []byte{AccelXOutH}, data); err != nil {
		return nil, err
	}

	return &AccelerometerData{
		X: int16(data[0])<<8 | int16(data[1]),
		Y: int16(data[2])<<8 | int16(data[3]),
		Z: int16(data[4])<<8 | int16(data[5]),
	}, nil
}

// ReadGyroscope reads the gyroscope values.
func (d *Device) ReadGyroscope() (*GyroscopeData, error) {
	data := make([]byte, 6)
	if err := d.bus.Tx(uint16(d.address), []byte{GyroXOutH}, data); err != nil {
		return nil, err
	}

	return &GyroscopeData{
		X: int16(data[0])<<8 | int16(data[1]),
		Y: int16(data[2])<<8 | int16(data[3]),
		Z: int16(data[4])<<8 | int16(data[5]),
	}, nil
}

// ReadTemperature reads the temperature sensor value in degrees Celsius.
func (d *Device) ReadTemperature() (float32, error) {
	data := make([]byte, 2)
	if err := d.bus.Tx(uint16(d.address), []byte{TempOutH}, data); err != nil {
		return 0, err
	}

	temp := int16(data[0])<<8 | int16(data[1])
	// Temperature in degrees C = (TEMP_OUT / 340) + 36.53
	return float32(temp)/340.0 + 36.53, nil
}

// Connected checks if the device is connected by reading the WhoAmI register.
func (d *Device) Connected() bool {
	whoami, err := d.read8(WhoAmI)
	return err == nil && whoami == WhoAmIValue
}

func (d *Device) write8(reg uint8, value uint8) error {
	return d.bus.Tx(uint16(d.address), []byte{reg, value}, nil)
}

func (d *Device) read8(reg uint8) (uint8, error) {
	data := make([]byte, 1)
	err := d.bus.Tx(uint16(d.address), []byte{reg}, data)
	return data[0], err
}
