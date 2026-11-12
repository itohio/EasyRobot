//go:build !tinygo

package devices

import "testing"

// The host I2C target events must keep TinyGo's machine.I2CTargetEvent
// values (I2CReceive 0, I2CRequest 1, I2CFinish 2), which i2c_tinygo.go
// aliases, so code switching on them behaves the same on both.
func TestI2CTargetEventValuesMatchTinyGo(t *testing.T) {
	if I2CReceive != 0 || I2CRequest != 1 || I2CFinish != 2 {
		t.Fatalf("I2CReceive %d, I2CRequest %d, I2CFinish %d; want 0, 1, 2", I2CReceive, I2CRequest, I2CFinish)
	}
}
