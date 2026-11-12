// Package mpu6050 drives the InvenSense MPU-6050 6-axis accelerometer and
// gyroscope over I2C.
//
// A read is one 14-byte burst from ACCEL_XOUT_H (accel, temperature, gyro of
// the same sample) into the device's own buffer; [Device.ReadRaw] and
// [Device.Read] fill caller-owned values, so reading allocates nothing (TinyGo
// friendly). [Config] sets the sample rate, the digital low-pass filter, the
// ranges and the data-ready interrupt; [Config.Scale] converts raw counts to
// m/s², rad/s and °C.
//
// Datasheet and register map: MPU-6000/MPU-6050 Product Specification rev 3.4
// (PS-MPU-6000A-00) and Register Map rev 4.2 (RM-MPU-6000A-00).
package mpu6050

import (
	"github.com/itohio/EasyRobot/x/devices"
)

// DefaultAddress is the I2C address with AD0 low; AD0 high gives 0x69.
const DefaultAddress = 0x68

// WhoAmIValue is WHO_AM_I's value on an MPU-6050 (whatever AD0 is).
const WhoAmIValue = 0x68

// Registers (register map rev 4.2).
const (
	RegSmplrtDiv   = 0x19
	RegConfig      = 0x1A
	RegGyroConfig  = 0x1B
	RegAccelConfig = 0x1C
	RegIntPinCfg   = 0x37
	RegIntEnable   = 0x38
	RegIntStatus   = 0x3A
	RegAccelXOutH  = 0x3B // accel x, y, z, temp, gyro x, y, z: 14 bytes, big-endian
	RegTempOutH    = 0x41
	RegGyroXOutH   = 0x43
	RegPwrMgmt1    = 0x6B
	RegWhoAmI      = 0x75
)

const (
	pwrClockPLLGyroX = 0x01 // PWR_MGMT_1: awake, CLKSEL=1 (more stable than the 8 MHz oscillator)
	intLatch         = 0x20 // INT_PIN_CFG LATCH_INT_EN: INT held high until cleared
	intReadClear     = 0x10 // INT_PIN_CFG INT_RD_CLEAR: any read clears it
	intDataReady     = 0x01 // INT_ENABLE DATA_RDY_EN, INT_STATUS DATA_RDY_INT
	burstLen         = 14
)

// DLPF is the digital low-pass filter setting (CONFIG DLPF_CFG); the
// bandwidths are accel / gyro.
type DLPF uint8

const (
	DLPFOff   DLPF = iota // 260 / 256 Hz; the gyro output rate is 8 kHz
	DLPF184Hz             // 184 / 188 Hz; from here on the gyro output rate is 1 kHz
	DLPF94Hz              // 94 / 98 Hz
	DLPF44Hz              // 44 / 42 Hz
	DLPF21Hz              // 21 / 20 Hz
	DLPF10Hz              // 10 / 10 Hz
	DLPF5Hz               // 5 / 5 Hz
)

// AccelRange is the accelerometer full scale (ACCEL_CONFIG AFS_SEL).
type AccelRange uint8

const (
	Accel2G  AccelRange = iota // 16384 LSB/g
	Accel4G                    // 8192 LSB/g
	Accel8G                    // 4096 LSB/g
	Accel16G                   // 2048 LSB/g
)

// GyroRange is the gyroscope full scale (GYRO_CONFIG FS_SEL).
type GyroRange uint8

const (
	Gyro250DPS  GyroRange = iota // 131 LSB/(°/s)
	Gyro500DPS                   // 65.5 LSB/(°/s)
	Gyro1000DPS                  // 32.8 LSB/(°/s)
	Gyro2000DPS                  // 16.4 LSB/(°/s)
)

const standardGravity = 9.80665 // m/s²

// accelScale and gyroScale are m/s² and rad/s per count, per range.
var (
	accelScale = [4]float32{standardGravity / 16384, standardGravity / 8192, standardGravity / 4096, standardGravity / 2048}
	gyroScale  = [4]float32{deg / 131, deg / 65.5, deg / 32.8, deg / 16.4}
)

const deg = 3.14159265358979323846 / 180

// Config is the measurement configuration. The zero value matches the chip's
// power-on ranges and filter: DLPF off, accel ±2 g, gyro ±250 °/s, divider 0
// (8 kHz gyro, 1 kHz accel), no interrupt. The chip powers up asleep, so
// reads return stale or zero data until Configure has woken it.
type Config struct {
	// SampleRateDiv: sample rate = gyro output rate / (1 + SampleRateDiv); the
	// gyro output rate is 8 kHz with DLPFOff and 1 kHz otherwise.
	SampleRateDiv uint8
	DLPF          DLPF
	Accel         AccelRange
	Gyro          GyroRange
	// DataReadyInterrupt drives INT high when a sample is ready, latched
	// until any register read (a sample read clears it), so both an edge
	// interrupt and a polled level work.
	DataReadyInterrupt bool
}

func (c Config) valid() bool {
	return c.DLPF <= DLPF5Hz && c.Accel <= Accel16G && c.Gyro <= Gyro2000DPS
}

// SampleRate is the sample rate in Hz.
func (c Config) SampleRate() float32 {
	rate := float32(1000)
	if c.DLPF == DLPFOff {
		rate = 8000
	}
	return rate / (1 + float32(c.SampleRateDiv))
}

// Raw is one sample in counts.
type Raw struct {
	Accel [3]int16
	Temp  int16
	Gyro  [3]int16
}

// Sample is one sample in SI units: accel m/s², gyro rad/s, temperature °C.
type Sample struct {
	Accel [3]float32
	Gyro  [3]float32
	TempC float32
}

// Scale converts r to SI units with c's ranges (temperature: counts/340 +
// 36.53 °C).
func (c Config) Scale(r Raw) Sample {
	a, g := accelScale[c.Accel&3], gyroScale[c.Gyro&3]
	return Sample{
		Accel: [3]float32{float32(r.Accel[0]) * a, float32(r.Accel[1]) * a, float32(r.Accel[2]) * a},
		Gyro:  [3]float32{float32(r.Gyro[0]) * g, float32(r.Gyro[1]) * g, float32(r.Gyro[2]) * g},
		TempC: float32(r.Temp)/340 + 36.53,
	}
}

// Device is an MPU-6050 on an I2C bus.
type Device struct {
	bus  devices.I2C
	addr uint16
	cfg  Config
	buf  [1 + burstLen]byte // register address + burst; reused by every transfer
}

// New returns a device at address (0 means [DefaultAddress]) on a configured
// bus: a TinyGo machine.I2C or a Linux bus from devices.NewI2C. It does not
// touch the bus; call [Device.Probe] and [Device.Configure].
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{bus: bus, addr: uint16(address)}
}

// Probe reads WHO_AM_I: id is what answered, ok whether it is an MPU-6050.
// Related chips answer other ids (an MPU-6500 0x70, an MPU-9250 0x71).
func (d *Device) Probe() (id uint8, ok bool, err error) {
	id, err = d.read8(RegWhoAmI)
	return id, err == nil && id == WhoAmIValue, err
}

// Configure wakes the device and applies c. An out-of-range field gives
// devices.ErrInvalidValue before anything is written. The writes are not
// atomic: on a bus error Configure stops, and the configuration it records
// (and Read scales with) is what landed before the failure, so the scaling
// always matches the ranges in the chip.
func (d *Device) Configure(c Config) error {
	if !c.valid() {
		return devices.ErrInvalidValue
	}
	var intEnable uint8
	if c.DataReadyInterrupt {
		intEnable = intDataReady
	}
	if err := d.write8(RegPwrMgmt1, pwrClockPLLGyroX); err != nil {
		return err
	}
	if err := d.write8(RegSmplrtDiv, c.SampleRateDiv); err != nil {
		return err
	}
	d.cfg.SampleRateDiv = c.SampleRateDiv
	if err := d.write8(RegConfig, uint8(c.DLPF)); err != nil {
		return err
	}
	d.cfg.DLPF = c.DLPF
	if err := d.write8(RegGyroConfig, uint8(c.Gyro)<<3); err != nil {
		return err
	}
	d.cfg.Gyro = c.Gyro
	if err := d.write8(RegAccelConfig, uint8(c.Accel)<<3); err != nil {
		return err
	}
	d.cfg.Accel = c.Accel
	if err := d.write8(RegIntPinCfg, intLatch|intReadClear); err != nil {
		return err
	}
	if err := d.write8(RegIntEnable, intEnable); err != nil {
		return err
	}
	d.cfg.DataReadyInterrupt = c.DataReadyInterrupt
	return nil
}

// Config is the configuration last applied (the zero value before
// [Device.Configure]). Configure writes only the registers in Config and
// PWR_MGMT_1; PWR_MGMT_2, USER_CTRL and FIFO_EN keep whatever they hold.
func (d *Device) Config() Config { return d.cfg }

// SampleRate is the configured sample rate in Hz.
func (d *Device) SampleRate() float32 { return d.cfg.SampleRate() }

// DataReady reports INT_STATUS DATA_RDY_INT (reading it clears it).
func (d *Device) DataReady() (bool, error) {
	s, err := d.read8(RegIntStatus)
	return s&intDataReady != 0, err
}

// ReadRaw reads one sample (accel, temperature, gyro) in a single burst.
func (d *Device) ReadRaw(r *Raw) error {
	d.buf[0] = RegAccelXOutH
	if err := d.bus.Tx(d.addr, d.buf[:1], d.buf[1:]); err != nil {
		return err
	}
	b := d.buf[1:]
	for i := 0; i < 3; i++ {
		r.Accel[i] = int16(uint16(b[2*i])<<8 | uint16(b[2*i+1]))
		r.Gyro[i] = int16(uint16(b[8+2*i])<<8 | uint16(b[9+2*i]))
	}
	r.Temp = int16(uint16(b[6])<<8 | uint16(b[7]))
	return nil
}

// Read reads one sample and scales it with the configured ranges.
func (d *Device) Read(s *Sample) error {
	var r Raw
	if err := d.ReadRaw(&r); err != nil {
		return err
	}
	*s = d.cfg.Scale(r)
	return nil
}

func (d *Device) write8(reg, v uint8) error {
	d.buf[0], d.buf[1] = reg, v
	return d.bus.Tx(d.addr, d.buf[:2], nil)
}

func (d *Device) read8(reg uint8) (uint8, error) {
	d.buf[0] = reg
	if err := d.bus.Tx(d.addr, d.buf[:1], d.buf[1:2]); err != nil {
		return 0, err // not a leftover byte from an earlier transfer
	}
	return d.buf[1], nil
}
