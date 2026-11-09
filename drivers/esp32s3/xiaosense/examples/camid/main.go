//go:build xiao_esp32s3

// Command camid starts the XIAO ESP32-S3 Sense camera clock and SCCB, reads
// the sensor id and prints it once a second on the USB serial port.
package main

import (
	"time"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/esp32s3/xiaosense"
)

func main() {
	for {
		id, err := camera.Identify(camera.Config{}, xiaosense.CameraPins)
		if err != nil {
			println("camid: err", err.Error())
		} else {
			println("camid: sensor", id.Model.String(), "addr", id.Addr, "pid", id.PID, "ver", id.VER)
		}
		time.Sleep(time.Second)
	}
}
