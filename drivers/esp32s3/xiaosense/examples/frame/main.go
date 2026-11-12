//go:build xiao_esp32s3

// Command frame streams camera images over the USB serial port in the camera
// frame format v1 (drivers/camera/frame), for cmd/camframe on the host.
//
// Build-time option (tinygo -ldflags "-X main.mode=jpeg-qxga"): mode is
// yuv-qvga (default), y8-qvga, rgb565-qvga, jpeg-xga or jpeg-qxga; crop=1
// takes a centred window instead of the full field; pattern=1 the sensor's
// colour bars. JPEG uses a 290 KB buffer: 2048×1536 needs quality ≥ 40 to fit
// (M3 evidence, outside the repo), so jpeg-qxga sets 40, jpeg-xga 12.
package main

import (
	"machine"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov3660"
	"github.com/itohio/EasyRobot/drivers/esp32s3/xiaosense"
)

var (
	mode    = "yuv-qvga"
	crop    = "0"
	pattern = "0"
)

const jpegBuf = 72 * 4032 // 290,304 B

func config() (camera.Config, frame.Format, uint8) {
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.QVGA, Format: sensor.YUV422}}
	f, q := frame.YUYV, uint8(0)
	switch mode {
	case "y8-qvga":
		cfg.Format, f = sensor.Y8, frame.Y8
	case "rgb565-qvga":
		cfg.Format, f = sensor.RGB565, frame.RGB565LE
	case "jpeg-xga":
		cfg.Resolution, cfg.Format, f, q = sensor.Resolution{W: 1024, H: 768}, sensor.JPEG, frame.JPEG, 12
	case "jpeg-qxga":
		cfg.Resolution, cfg.Format, f, q = sensor.Resolution{W: 2048, H: 1536}, sensor.JPEG, frame.JPEG, 40
	}
	cfg.Crop = crop == "1"
	cfg.TestPattern = pattern == "1"
	return cfg, f, q
}

// align4 returns a 4-byte aligned window of n bytes (GDMA descriptors need
// word-aligned buffers).
func align4(n int) []byte {
	b := make([]byte, n+4)
	off := int(-uintptr(unsafe.Pointer(&b[0])) & 3)
	return b[off : off+n]
}

func main() {
	time.Sleep(2 * time.Second) // let the host open the port
	cfg, f, q := config()
	println("frame: mode", mode)
	cam, err := camera.Open(cfg, xiaosense.CameraPins)
	for err != nil {
		println("frame: open:", err.Error())
		time.Sleep(time.Second)
		cam, err = camera.Open(cfg, xiaosense.CameraPins)
	}
	id := cam.Sensor()
	println("frame: sensor", id.Model.String())
	if s, ok := cam.Driver().(*ov3660.Sensor); ok && q != 0 {
		if err := s.SetQuality(q); err != nil {
			println("frame: quality:", err.Error())
		}
	}

	n := camera.FrameSize(cfg)
	if cfg.Format == sensor.JPEG {
		n = min(camera.JPEGBufSize(cfg), jpegBuf) // internal SRAM: at most 290 KB next to the rest
	}
	buf := align4(n)
	enc := frame.NewEncoder(machine.Serial)
	meta := frame.Meta{Width: cfg.W, Height: cfg.H, Format: f, Sensor: frame.Sensor(id.Model)}

	for range 10 { // let auto exposure settle
		if _, err := cam.Capture(buf); err != nil {
			println("frame: warm-up capture:", err.Error())
		}
	}
	start := time.Now()
	for {
		t0 := time.Now()
		fr, err := cam.Capture(buf)
		if err != nil {
			println("frame: capture:", err.Error())
			continue
		}
		ts := uint32(t0.Sub(start).Microseconds())
		if err := enc.WriteImage(meta, ts, fr.Data); err != nil {
			println("frame: write:", err.Error())
		}
	}
}
