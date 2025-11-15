//go:build !tinygo

package servo

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/devices"
)

// newServoArray creates a stub array for non-TinyGo platforms.
func newServoArray(pwm devices.PWMDevice, motors []Motor) (*ServoArray, error) {
	// For non-TinyGo platforms, create a stub array
	// TODO: Implement Linux PWM support if needed
	return nil, fmt.Errorf("PWM servo array not implemented for this platform")
}

