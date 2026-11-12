//go:build !tinygo

package devices

// I2CTargetEvent is an I2C target-mode event; the values and names are
// TinyGo's machine.I2CTargetEvent (aliased there, see i2c_tinygo.go).
type I2CTargetEvent uint8

const (
	// I2CReceive: the target received a message from the controller.
	I2CReceive I2CTargetEvent = iota
	// I2CRequest: the controller expects a message from the target.
	I2CRequest
	// I2CFinish: the controller ended the transaction with a stop.
	I2CFinish
)
