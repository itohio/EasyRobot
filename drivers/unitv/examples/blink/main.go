//go:build unitv

// Command blink toggles the WS2812 data pin (G8) at about 1 Hz: a GPIO build
// smoke for the UnitV target. G8 feeds a WS2812, which needs timed pulses, so
// this shows no defined colour. It uses a bare time.Sleep, which can return
// early on TinyGo 0.43/K210 (ER-K210-M1 evidence §4, outside the repo), so the toggle
// is irregular; that is harmless for a build smoke.
package main

import (
	"machine"
	"time"

	"github.com/itohio/EasyRobot/drivers/unitv"
)

func main() {
	led := unitv.WS2812
	led.Configure(machine.PinConfig{Mode: machine.PinOutput})
	for {
		led.Low()
		time.Sleep(500 * time.Millisecond)
		led.High()
		time.Sleep(500 * time.Millisecond)
	}
}
