//go:build xiao_esp32s3

// Package xiaosense holds the pin map of the Seeed Studio XIAO ESP32-S3
// Sense: the XIAO ESP32-S3 with its camera/microSD expansion board (OV3660
// on the units here; OV2640 on earlier ones).
package xiaosense

import (
	"machine"

	"github.com/itohio/EasyRobot/drivers/camera"
)

// CameraPins are the expansion board's DVP camera pins (Seeed wiki; verified
// by the E2 sensor-id read and the E3 frame, ER-ESP32CAM evidence, outside
// the repo). SCCB is on I2C0, XCLK on LEDC timer 0, capture on GDMA channel 1
// (channel 0 is left to other drivers).
var CameraPins = camera.Pins{
	D:     [8]machine.Pin{machine.GPIO15, machine.GPIO17, machine.GPIO18, machine.GPIO16, machine.GPIO14, machine.GPIO12, machine.GPIO11, machine.GPIO48},
	PCLK:  machine.GPIO13,
	VSYNC: machine.GPIO38,
	HREF:  machine.GPIO47,
	XCLK:  machine.GPIO10,
	SDA:   machine.GPIO40,
	SCL:   machine.GPIO39,
	DMA:   1,
}

// microSD slot (SPI; CS is shared with the user LED's GPIO21).
const (
	SDSCK  = machine.GPIO7
	SDMISO = machine.GPIO8
	SDMOSI = machine.GPIO9
	SDCS   = machine.GPIO21
)
