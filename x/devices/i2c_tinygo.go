//go:build tinygo

package devices

import "machine"

// machine.I2C is used directly as I2C (and, where the port has target mode,
// I2CTarget).

// I2CTargetEvent is an I2C target-mode event.
type I2CTargetEvent = machine.I2CTargetEvent

const (
	I2CReceive = machine.I2CReceive
	I2CRequest = machine.I2CRequest
	I2CFinish  = machine.I2CFinish
)
