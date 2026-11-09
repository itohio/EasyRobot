//go:build unitv

// Command colorbar captures one QQVGA RGB565 frame of the OV2640's colour-bar
// test pattern through the DVP and prints rows 10, 60 and 110 as hex at
// 115200: per 20-pixel bar the centre pixel and the min/max of R, G, B
// (5/6/5-bit levels) over the bar's 12 interior pixels. It repeats every 3 s
// (ER-CAM-steps K1).
package main

import (
	"machine"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/unitv"
)

const hexDigits = "0123456789ABCDEF"

func hex16(v uint16) string {
	return string([]byte{hexDigits[v>>12], hexDigits[v>>8&15], hexDigits[v>>4&15], hexDigits[v&15]})
}

func main() {
	machine.UART0.Configure(machine.UARTConfig{BaudRate: 115200})
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.RGB565, TestPattern: true}}
	cam, err := camera.Open(cfg, unitv.CameraPins)
	if err != nil {
		println("colorbar: open:", err.Error())
		return
	}
	id := cam.Sensor()
	println("colorbar: sensor", id.Model.String(), "addr 0x30 pid", id.PID, "ver", id.VER)

	n := camera.FrameSize(cfg)
	raw := make([]byte, n+8)
	off := int(-uintptr(unsafe.Pointer(&raw[0])) & 7)
	buf := raw[off : off+n]
	for {
		fr, err := cam.Capture(buf)
		if err != nil {
			println("colorbar: capture:", err.Error())
		} else {
			px := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(fr.Data))), int(fr.Width)*int(fr.Height))
			println("colorbar: frame", fr.Seq, fr.Width, "x", fr.Height)
			for _, y := range [3]int{10, 60, 110} {
				print("row ", y, ":")
				for bar := 0; bar < 8; bar++ {
					var rmin, gmin, bmin uint16 = 31, 63, 31
					var rmax, gmax, bmax uint16
					for x := bar*20 + 4; x < bar*20+16; x++ { // interior: edges blend with neighbours
						p := px[y*int(fr.Width)+x]
						r, g, b := p>>11, p>>5&63, p&31
						rmin, rmax = min(rmin, r), max(rmax, r)
						gmin, gmax = min(gmin, g), max(gmax, g)
						bmin, bmax = min(bmin, b), max(bmax, b)
					}
					c := px[y*int(fr.Width)+bar*20+10]
					print(" ", hex16(c), "[r", rmin, "-", rmax, " g", gmin, "-", gmax, " b", bmin, "-", bmax, "]")
				}
				println()
			}
			print("row 60 first 16 px:")
			for x := range 16 {
				print(" ", hex16(px[60*int(fr.Width)+x]))
			}
			println()
		}
		t := time.Now()
		for time.Since(t) < 3*time.Second {
		}
	}
}
