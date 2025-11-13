//go:build tinygo

package tca9548a_test

import (
	"machine"

	"github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/tca9548a"
	"github.com/itohio/EasyRobot/x/devices/vl53l0x"
)

// ExampleRouter demonstrates how to use the TCA9548A router to access
// multiple I2C devices on different channels.
func ExampleRouter() {
	// Initialize the base I2C bus
	machineI2C := machine.I2C0
	machineI2C.Configure(machine.I2CConfig{
		Frequency: machine.TWI_FREQ_400KHZ,
	})

	// Wrap machine.I2C with the abstraction
	bus := devices.NewTinyGoI2C(machineI2C)

	// Create a router for the TCA9548A at address 0x70
	router := tca9548a.NewRouter(bus, 0x70)
	router.Configure(true) // Initialize and reset all channels

	// Get I2C interfaces for different channels
	// These can be used directly with any I2C device driver
	channel0, _ := router.Channel(0)
	channel1, _ := router.Channel(1)
	channel2, _ := router.Channel(2)

	// Use the channel interfaces with device drivers
	// Each device will automatically use the correct channel
	sensor0 := vl53l0x.New(channel0, 0x29)
	sensor0.Configure(true)

	sensor1 := vl53l0x.New(channel1, 0x29)
	sensor1.Configure(true)

	sensor2 := vl53l0x.New(channel2, 0x29)
	sensor2.Configure(true)

	// Now you can use the sensors - each will automatically
	// route through the correct TCA9548A channel
	_, _ = sensor0.ReadRangeSingle()
	_, _ = sensor1.ReadRangeSingle()
	_, _ = sensor2.ReadRangeSingle()
}
