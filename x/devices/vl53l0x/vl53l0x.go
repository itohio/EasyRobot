// Package vl53l0x provides a driver for the VL53L0X time-of-flight distance sensor.
//
// This is a simplified implementation. For full functionality, refer to the STMicroelectronics
// VL53L0X API documentation.
//
// Datasheet: https://www.st.com/resource/en/datasheet/vl53l0x.pdf
package vl53l0x

import (
	"time"

	"github.com/itohio/EasyRobot/x/devices"
)

// DefaultAddress is the default I2C address for the VL53L0X
const DefaultAddress = 0x29

// Register addresses (key registers only - full list in datasheet)
const (
	SysRangeStart                        = 0x00
	SystemSequenceConfig                 = 0x01
	SystemIntermeasurementPeriod         = 0x04
	SystemThresholdHigh                  = 0x0C
	SystemThresholdLow                   = 0x0E
	SystemRangeConfig                    = 0x09
	SystemInterruptConfigGPIO            = 0x0A
	GPIOHvMuxActiveHigh                  = 0x84
	SystemInterruptClear                 = 0x0B
	ResultInterruptStatus                = 0x13
	ResultRangeStatus                    = 0x14
	ResultCoreAmbientWindowEventsRtn     = 0xBC
	ResultCoreRangingTotalEventsRtn      = 0xC0
	ResultCoreAmbientWindowEventsRef     = 0xD0
	ResultCoreRangingTotalEventsRef      = 0xD4
	ResultPeakSignalRateRef              = 0xB6
	AlgoPartToPartRangeOffsetMM          = 0x28
	I2CSlaveDeviceAddress                = 0x8A
	MsrcConfigControl                    = 0x60
	PreRangeConfigMinSNR                 = 0x27
	PreRangeConfigValidPhaseLow          = 0x56
	PreRangeConfigValidPhaseHigh         = 0x57
	PreRangeMinCountRateRtnLimit         = 0x64
	FinalRangeConfigMinSNR               = 0x67
	FinalRangeConfigValidPhaseLow        = 0x47
	FinalRangeConfigValidPhaseHigh       = 0x48
	FinalRangeConfigMinCountRateRtnLimit = 0x44
	PreRangeConfigSigmaThreshHi          = 0x61
	PreRangeConfigSigmaThreshLo          = 0x62
	PreRangeConfigVcselPeriod            = 0x50
	PreRangeConfigTimeoutMacropHi        = 0x51
	PreRangeConfigTimeoutMacropLo        = 0x52
	SystemHistogramBin                   = 0x81
	HistogramConfigInitialPhaseSelect    = 0x33
	HistogramConfigReadoutCtrl           = 0x55
	FinalRangeConfigVcselPeriod          = 0x70
	FinalRangeConfigTimeoutMacropHi      = 0x71
	FinalRangeConfigTimeoutMacropLo      = 0x72
	CrosstalkCompensationPeakRateMcps    = 0x20
	MsrcConfigTimeoutMacrop              = 0x46
	SoftResetGo2SoftResetN               = 0xBF
	IdentificationModelID                = 0xC0
	IdentificationRevisionID             = 0xC2
	OscCalibrateVal                      = 0xF8
	GlobalConfigVcselWidth               = 0x32
	GlobalConfigSpadEnablesRef0          = 0xB0
	GlobalConfigSpadEnablesRef1          = 0xB1
	GlobalConfigSpadEnablesRef2          = 0xB2
	GlobalConfigSpadEnablesRef3          = 0xB3
	GlobalConfigSpadEnablesRef4          = 0xB4
	GlobalConfigSpadEnablesRef5          = 0xB5
	GlobalConfigRefEnStartSelect         = 0xB6
	DynamicSpadNumRequestedRefSpad       = 0x4E
	DynamicSpadRefEnStartOffset          = 0x4F
	PowerManagementGo1PowerForce         = 0x80
	VhvConfigPadSclSdaExtsupHv           = 0x89
	AlgoPhasecalLimit                    = 0x30
	AlgoPhasecalConfigTimeout            = 0x30
)

// VcselPeriodType represents the type of VCSEL period
type VcselPeriodType uint8

const (
	VcselPeriodPreRange VcselPeriodType = iota
	VcselPeriodFinalRange
)

// Device wraps an I2C connection to a VL53L0X device.
type Device struct {
	bus                     devices.I2C
	address                 uint8
	stopVariable            uint8
	measurementTimingBudget uint32
	ioTimeout               time.Duration
	continuousMeasurement   bool
	measurementInitiated    bool
}

// New creates a new VL53L0X connection. The I2C bus must already be configured.
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{
		bus:       bus,
		address:   address,
		ioTimeout: 0, // 0 means no timeout
	}
}

// Configure initializes the device. If init is true, performs full initialization.
func (d *Device) Configure(init bool) error {
	if init {
		return d.Reset()
	}
	return nil
}

// Reset performs a full reset and initialization of the device.
func (d *Device) Reset() error {
	d.measurementInitiated = false
	d.continuousMeasurement = false
	return d.init(true) // true = 2.8V I/O mode
}

// Connected checks if the device is connected by reading identification registers.
func (d *Device) Connected() bool {
	modelID, err1 := d.read8(IdentificationModelID)
	revID, err2 := d.read8(IdentificationRevisionID)
	// Expected: Model ID = 0xEE, Rev ID = 0xAA, Rev ID MSB = 0x10
	return err1 == nil && err2 == nil && modelID == 0xEE && revID == 0xAA
}

// ReadRangeSingle performs a single range measurement and returns the distance in millimeters.
func (d *Device) ReadRangeSingle() (uint16, error) {
	if err := d.startRangeSingle(); err != nil {
		return 0, err
	}

	// Wait for measurement to complete
	timeout := time.Now().Add(100 * time.Millisecond)
	for time.Now().Before(timeout) {
		status, err := d.read8(ResultInterruptStatus)
		if err != nil {
			return 0, err
		}
		if (status & 0x07) != 0 {
			break
		}
		time.Sleep(1 * time.Millisecond)
	}

	// Read range
	data := make([]byte, 2)
	if err := d.bus.Tx(uint16(d.address), []byte{ResultRangeStatus + 10}, data); err != nil {
		return 0, err
	}

	rangeMM := uint16(data[0])<<8 | uint16(data[1])

	// Clear interrupt
	if err := d.write8(SystemInterruptClear, 0x01); err != nil {
		return rangeMM, err
	}

	d.measurementInitiated = false
	return rangeMM, nil
}

// StartContinuous starts continuous ranging measurements.
func (d *Device) StartContinuous(periodMs uint32) error {
	// Simplified continuous start - full implementation would handle timing budgets
	if err := d.write8(SysRangeStart, 0x02); err != nil { // Back-to-back mode
		return err
	}
	d.continuousMeasurement = true
	d.measurementInitiated = true
	return nil
}

// StopContinuous stops continuous measurements.
func (d *Device) StopContinuous() error {
	if err := d.write8(SysRangeStart, 0x01); err != nil { // Single shot mode
		return err
	}
	d.continuousMeasurement = false
	d.measurementInitiated = false
	return nil
}

// ReadRangeContinuous reads a range measurement during continuous mode.
func (d *Device) ReadRangeContinuous() (uint16, error) {
	if !d.continuousMeasurement {
		return 0, devices.ErrInvalidState
	}

	// Check if measurement is ready
	status, err := d.read8(ResultInterruptStatus)
	if err != nil {
		return 0, err
	}
	if (status & 0x07) == 0 {
		return 0, devices.ErrTimeout
	}

	// Read range
	data := make([]byte, 2)
	if err := d.bus.Tx(uint16(d.address), []byte{ResultRangeStatus + 10}, data); err != nil {
		return 0, err
	}

	rangeMM := uint16(data[0])<<8 | uint16(data[1])

	// Clear interrupt
	if err := d.write8(SystemInterruptClear, 0x01); err != nil {
		return rangeMM, err
	}

	return rangeMM, nil
}

// init performs the full initialization sequence (simplified version).
func (d *Device) init(io2v8 bool) error {
	// Enable 2.8V I/O if requested
	if io2v8 {
		val, err := d.read8(VhvConfigPadSclSdaExtsupHv)
		if err != nil {
			return err
		}
		if err := d.write8(VhvConfigPadSclSdaExtsupHv, val|0x01); err != nil {
			return err
		}
	}

	// Set I2C standard mode
	if err := d.write8(0x88, 0x00); err != nil {
		return err
	}

	// Get stop variable
	if err := d.write8(0x80, 0x01); err != nil {
		return err
	}
	if err := d.write8(0xFF, 0x01); err != nil {
		return err
	}
	if err := d.write8(0x00, 0x00); err != nil {
		return err
	}
	stopVar, err := d.read8(0x91)
	if err != nil {
		return err
	}
	d.stopVariable = stopVar
	if err := d.write8(0x00, 0x01); err != nil {
		return err
	}
	if err := d.write8(0xFF, 0x00); err != nil {
		return err
	}
	if err := d.write8(0x80, 0x00); err != nil {
		return err
	}

	// Disable MSRC and TCC limit checks
	val, err := d.read8(MsrcConfigControl)
	if err != nil {
		return err
	}
	if err := d.write8(MsrcConfigControl, val|0x12); err != nil {
		return err
	}

	// Set signal rate limit
	if err := d.SetSignalRateLimit(0.25); err != nil {
		return err
	}

	// Set sequence config
	if err := d.write8(SystemSequenceConfig, 0xFF); err != nil {
		return err
	}

	// Load tuning data
	for _, tuning := range TuningData {
		if err := d.write8(tuning.Reg, tuning.Value); err != nil {
			return err
		}
	}

	// Configure interrupt
	if err := d.write8(SystemInterruptConfigGPIO, 0x04); err != nil {
		return err
	}
	val, err = d.read8(GPIOHvMuxActiveHigh)
	if err != nil {
		return err
	}
	if err := d.write8(GPIOHvMuxActiveHigh, val&^0x10); err != nil {
		return err
	}
	if err := d.write8(SystemInterruptClear, 0x01); err != nil {
		return err
	}

	// Set sequence config for normal operation
	if err := d.write8(SystemSequenceConfig, 0xE8); err != nil {
		return err
	}

	// Perform calibration
	return d.Calibrate()
}

// Calibrate performs reference calibration.
func (d *Device) Calibrate() error {
	// Simplified calibration - full implementation would perform full calibration sequence
	if err := d.write8(SystemSequenceConfig, 0x01); err != nil {
		return err
	}
	// Perform calibration steps...
	if err := d.write8(SystemSequenceConfig, 0x02); err != nil {
		return err
	}
	// More calibration steps...
	if err := d.write8(SystemSequenceConfig, 0xE8); err != nil {
		return err
	}
	return nil
}

// SetSignalRateLimit sets the signal rate limit in MCPS.
func (d *Device) SetSignalRateLimit(limitMcps float32) error {
	if limitMcps < 0 || limitMcps > 511.99 {
		return devices.ErrInvalidValue
	}
	// Q9.7 fixed point format
	limitFixed := uint16(limitMcps * 128)
	return d.write16(FinalRangeConfigMinCountRateRtnLimit, limitFixed)
}

// startRangeSingle starts a single range measurement.
func (d *Device) startRangeSingle() error {
	if err := d.write8(0x80, 0x01); err != nil {
		return err
	}
	if err := d.write8(0xFF, 0x01); err != nil {
		return err
	}
	if err := d.write8(0x00, 0x00); err != nil {
		return err
	}
	if err := d.write8(0x91, d.stopVariable); err != nil {
		return err
	}
	if err := d.write8(0x00, 0x01); err != nil {
		return err
	}
	if err := d.write8(0xFF, 0x00); err != nil {
		return err
	}
	if err := d.write8(0x80, 0x00); err != nil {
		return err
	}
	if err := d.write8(SysRangeStart, 0x01); err != nil {
		return err
	}

	d.measurementInitiated = true
	d.continuousMeasurement = false

	// Wait for start bit to clear
	timeout := time.Now().Add(100 * time.Millisecond)
	for time.Now().Before(timeout) {
		val, err := d.read8(SysRangeStart)
		if err != nil {
			return err
		}
		if (val & 0x01) == 0 {
			return nil
		}
		time.Sleep(1 * time.Millisecond)
	}
	return devices.ErrTimeout
}

func (d *Device) write8(reg uint8, value uint8) error {
	return d.bus.Tx(uint16(d.address), []byte{reg, value}, nil)
}

func (d *Device) read8(reg uint8) (uint8, error) {
	data := make([]byte, 1)
	err := d.bus.Tx(uint16(d.address), []byte{reg}, data)
	return data[0], err
}

func (d *Device) write16(reg uint8, value uint16) error {
	return d.bus.Tx(uint16(d.address), []byte{reg, byte(value >> 8), byte(value & 0xFF)}, nil)
}
