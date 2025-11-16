package encoder_test

import (
	"time"

	"github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/encoder"
)

// ExampleEncoder demonstrates how to use a quadrature encoder.
func ExampleEncoder() {
	// Define encoder pins (A and B channels)
	pinA, err := devices.NewPin(2)
	if err != nil {
		// handle error
		return
	}
	pinB, err := devices.NewPin(3)
	if err != nil {
		// handle error
		return
	}

	// Create encoder with default configuration
	// Default: 2048 counts per revolution (512 PPR * 4x decoding)
	dev := encoder.New(pinA, pinB, encoder.DefaultConfig())

	// Or use custom configuration
	// dev := encoder.New(pinA, pinB, encoder.Config{
	// 	CountsPerRevolution: 4096, // 1024 PPR * 4x decoding
	// 	UpdateInterval:     50 * time.Millisecond,
	// })

	// Configure the encoder (sets up pins and interrupts)
	if err := dev.Configure(); err != nil {
		// handle error
		return
	}

	// Reset position to zero
	dev.Reset()

	// In your main loop, periodically read position and RPM
	for {
		// Read current position in counts
		position := dev.Position()
		println("Position:", position)

		// Read current RPM
		rpm := dev.RPM()
		println("RPM:", rpm)

		// Calculate revolutions from position
		revolutions := float64(position) / float64(dev.CountsPerRevolution())
		println("Revolutions:", revolutions)

		time.Sleep(100 * time.Millisecond)
	}
}
