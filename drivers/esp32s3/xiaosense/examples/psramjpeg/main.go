//go:build xiao_esp32s3

// Command psramjpeg captures full-resolution OV3660 JPEGs (2048×1536) by
// GDMA straight into a PSRAM buffer and streams them over the USB serial
// port in the camera frame format v1, for cmd/camframe on the host (E7).
//
//	tinygo build -target=xiao-esp32s3 ./drivers/esp32s3/xiaosense/examples/psramjpeg
//
// Option (tinygo -ldflags "-X main.quality=6"): JPEG quality 0 (best) … 63,
// default 10. A 1.5 MB PSRAM buffer holds what internal RAM cannot (internal
// RAM fits 2048×1536 only at quality ≥ 40, M3 evidence).
package main

import (
	"machine"
	"strconv"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov3660"
	"github.com/itohio/EasyRobot/drivers/esp32s3/psram"
	"github.com/itohio/EasyRobot/drivers/esp32s3/xiaosense"
)

var quality = "10"

func main() {
	time.Sleep(2 * time.Second) // let the host open the port
	r, regs, err := psram.Init()
	if err != nil {
		for {
			println("psramjpeg: psram:", err.Error(), "vendor", regs.VendorID())
			time.Sleep(time.Second)
		}
	}
	arena := psram.NewArena(r)
	buf, err := arena.Alloc(camera.MaxJPEG)
	if err != nil {
		for {
			println("psramjpeg: alloc:", err.Error())
			time.Sleep(time.Second)
		}
	}
	println("psramjpeg: psram", r.Size(), "B; buffer", len(buf), "B at", uintptr(unsafe.Pointer(&buf[0])))

	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.Resolution{W: 2048, H: 1536}, Format: sensor.JPEG}}
	cam, err := camera.Open(cfg, xiaosense.CameraPins)
	for err != nil {
		println("psramjpeg: open:", err.Error())
		time.Sleep(time.Second)
		cam, err = camera.Open(cfg, xiaosense.CameraPins)
	}
	cam.SetCacheSync(psram.Cache{})
	q, _ := strconv.Atoi(quality)
	if s, ok := cam.Driver().(*ov3660.Sensor); ok {
		if err := s.SetQuality(uint8(q)); err != nil {
			println("psramjpeg: quality:", err.Error())
		}
	}
	println("psramjpeg: sensor", cam.Sensor().Model.String(), "2048x1536 JPEG quality", q)

	enc := frame.NewEncoder(machine.Serial)
	meta := frame.Meta{Width: cfg.W, Height: cfg.H, Format: frame.JPEG, Sensor: frame.Sensor(cam.Sensor().Model)}
	for range 10 { // let auto exposure settle
		if _, err := cam.Capture(buf); err != nil {
			println("psramjpeg: warm-up capture:", err.Error())
		}
	}
	start := time.Now()
	for {
		t0 := time.Now()
		fr, err := cam.Capture(buf)
		if err != nil {
			println("psramjpeg: capture:", err.Error())
			continue
		}
		capMs := time.Since(t0).Milliseconds()
		println("psramjpeg: seq", fr.Seq, "jpeg", len(fr.Data), "B at", uintptr(unsafe.Pointer(&fr.Data[0])), "capture ms", capMs)
		if err := enc.WriteImage(meta, uint32(t0.Sub(start).Microseconds()), fr.Data); err != nil {
			println("psramjpeg: write:", err.Error())
		}
	}
}
