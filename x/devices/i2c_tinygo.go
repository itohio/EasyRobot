//go:build tinygo

package devices

import "machine"

// Use device-specific I2C directly

// I2CTargetEvent represents an I2C target mode event.
type I2CTargetEvent = machine.I2CTargetEvent

const (
	I2CTargetEventStart   = machine.I2CTargetEventStart
	I2CTargetEventStop    = machine.I2CTargetEventStop
	I2CTargetEventAddress = machine.I2CTargetEventAddress
	I2CTargetEventData    = machine.I2CTargetEventData
	I2CTargetEventError   = machine.I2CTargetEventError
)
