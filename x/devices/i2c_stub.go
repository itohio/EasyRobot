//go:build !tinygo && !linux

package devices

// I2CTargetEvent represents an I2C target mode event.
type I2CTargetEvent uint8

const (
	I2CTargetEventStart I2CTargetEvent = iota
	I2CTargetEventStop
	I2CTargetEventAddress
	I2CTargetEventData
	I2CTargetEventError
)
