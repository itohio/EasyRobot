//go:build xiao_esp32s3

// Command controls exercises the OV3660 exposure, gain and white-balance
// controls on the XIAO ESP32-S3 Sense: auto for 3 s, then the auto values
// frozen as manual settings, then a manual exposure/gain step, printing the
// read-back each time (C1 controls). It also keeps SetControls in a built
// example: TinyGo 0.43 at -opt=z once failed to compile it (review C1).
package main

import (
	"time"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/esp32s3/xiaosense"
)

func show(label string, c sensor.Controls) {
	println("controls:", label, "ae", c.AutoExposure, "exp_us", c.ExposureUS, "agc", c.AutoGain,
		"gain_x100", int(c.Gain*100), "awb", c.AutoWB,
		"wb_x100", int(c.WB[0]*100), int(c.WB[1]*100), int(c.WB[2]*100))
}

func main() {
	time.Sleep(2 * time.Second)
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.QVGA, Format: sensor.Y8}}
	cam, err := camera.Open(cfg, xiaosense.CameraPins)
	for err != nil {
		println("controls: open:", err.Error())
		time.Sleep(time.Second)
		cam, err = camera.Open(cfg, xiaosense.CameraPins)
	}
	buf := make([]byte, camera.FrameSize(cfg))
	capture := func(n int) {
		for range n {
			if _, err := cam.Capture(buf); err != nil {
				println("controls: capture:", err.Error())
			}
		}
	}
	capture(30) // auto loops settle
	c, err := cam.ReadControls()
	if err != nil {
		println("controls: read:", err.Error())
	}
	show("auto", c)

	c.AutoExposure, c.AutoGain, c.AutoWB = false, false, false // freeze what the loops chose
	if err := cam.SetControls(c); err != nil {
		println("controls: set:", err.Error())
	}
	capture(5)
	if c, err = cam.ReadControls(); err == nil {
		show("frozen", c)
	}

	c.ExposureUS /= 2
	c.Gain *= 2
	if err := cam.SetControls(c); err != nil {
		println("controls: set:", err.Error())
	}
	capture(5)
	if c, err = cam.ReadControls(); err == nil {
		show("half exposure, double gain", c)
	}
	for {
		time.Sleep(time.Second)
	}
}
