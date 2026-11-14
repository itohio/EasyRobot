//go:build xiao_esp32s3

// Command webcam turns the XIAO ESP32-S3 Sense into a USB "webcam": on each
// request it captures one OV3660 JPEG, up to 2048×1536, by GDMA straight
// into a PSRAM buffer and sends it over the USB serial port in the camera
// frame format v1 (drivers/camera/frame). No radio.
//
// Request (one text line): "G <size> <quality>\n"
//
//	size:    0 = 2048×1536, 1 = 1600×1200, 2 = 1024×768, 3 = 800×600, 4 = 640×480
//	quality: OV3660 JPEG quantisation 25 … 63 (smaller = better); values
//	         below 25 are raised to 25, because lower ones corrupt dark frames
//	         (ER-ESP32CAM-E7 evidence, outside the repo)
//
// Reply: one frame-v1 image (chunks with CRC-32, data_len = the exact JPEG
// length), then the text line "webcam: sent <seq> <bytes> <ms>". Errors are
// text lines starting "webcam: error". Auto exposure runs on the sensor; the
// first capture after a size change waits for 8 settling frames.
//
// Host: cmd/camframe can request and save images, e.g.
//
//	camframe -port <port> -trigger $'G 0 30\n' -n 1 -out DIR
//
// Build with the stock target:
//
//	tinygo build -target=xiao-esp32s3 ./drivers/esp32s3/xiaosense/examples/webcam
package main

import (
	"machine"
	"strconv"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov3660"
	"github.com/itohio/EasyRobot/drivers/esp32s3/psram"
	"github.com/itohio/EasyRobot/drivers/esp32s3/xiaosense"
)

var sizes = [...]sensor.Resolution{{W: 2048, H: 1536}, {W: 1600, H: 1200}, {W: 1024, H: 768}, {W: 800, H: 600}, {W: 640, H: 480}}

const minQuality = 25

var (
	line [32]byte
	n    int
)

// readRequest returns the size index and quality of the next "G s q" line.
func readRequest() (int, uint8) {
	for {
		for machine.Serial.Buffered() == 0 {
			time.Sleep(2 * time.Millisecond)
		}
		b, _ := machine.Serial.ReadByte()
		if b != '\n' && b != '\r' {
			if n < len(line) {
				line[n] = b
				n++
			}
			continue
		}
		req := line[:n]
		n = 0
		if len(req) == 0 || req[0] != 'G' {
			continue
		}
		s, q := 0, 30
		f := fields(req[1:])
		if len(f) > 0 {
			if v, err := strconv.Atoi(f[0]); err == nil && v >= 0 && v < len(sizes) {
				s = v
			}
		}
		if len(f) > 1 {
			if v, err := strconv.Atoi(f[1]); err == nil && v >= 0 && v <= 63 {
				q = v
			}
		}
		if q < minQuality {
			q = minQuality
		}
		return s, uint8(q)
	}
}

func fields(b []byte) []string {
	var out []string
	start := -1
	for i, c := range b {
		if c == ' ' {
			if start >= 0 {
				out = append(out, string(b[start:i]))
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, string(b[start:]))
	}
	return out
}

// usbWriter writes to the USB serial port, retrying while the host drains
// the FIFO: TinyGo 0.43's USB-Serial/JTAG Write drops bytes once the FIFO
// stays full for ~50,000 polls, which tears chunks (CRC failures on the
// host). It gives up after 500 ms without progress.
type usbWriter struct{}

func (usbWriter) Write(p []byte) (int, error) {
	done := 0
	deadline := time.Now().Add(500 * time.Millisecond)
	for done < len(p) {
		k, err := machine.Serial.Write(p[done:])
		done += k
		if err == nil {
			continue
		}
		if k > 0 {
			deadline = time.Now().Add(500 * time.Millisecond)
		}
		if time.Now().After(deadline) {
			return done, err
		}
		time.Sleep(time.Millisecond)
	}
	return done, nil
}

func open(s int) (*camera.Camera, camera.Config, error) {
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sizes[s], Format: sensor.JPEG}}
	cam, err := camera.Open(cfg, xiaosense.CameraPins)
	if err != nil {
		return nil, cfg, err
	}
	cam.SetCacheSync(psram.Cache{})
	return cam, cfg, nil
}

func main() {
	time.Sleep(time.Second)
	r, _, err := psram.Init()
	if err != nil {
		for {
			println("webcam: error psram", err.Error())
			time.Sleep(time.Second)
		}
	}
	arena := psram.NewArena(r)
	buf, err := arena.Alloc(camera.JPEGBufSize(camera.Config{Mode: sensor.Mode{Resolution: sizes[0], Format: sensor.JPEG}}))
	if err != nil {
		for {
			println("webcam: error alloc", err.Error())
			time.Sleep(time.Second)
		}
	}
	enc := frame.NewEncoder(usbWriter{})
	var cam *camera.Camera
	var cfg camera.Config
	cur, settle := -1, 0
	println("webcam: ready; request \"G <size 0-4> <quality 25-63>\"")
	for {
		s, q := readRequest()
		if s != cur || cam == nil {
			if cam != nil {
				cam.Close()
			}
			if cam, cfg, err = open(s); err != nil {
				println("webcam: error open", err.Error())
				cam, cur = nil, -1
				continue
			}
			cur, settle = s, 8
		}
		if d, ok := cam.Driver().(*ov3660.Sensor); ok {
			if err := d.SetQuality(q); err != nil {
				println("webcam: error quality", err.Error())
			}
		}
		for ; settle > 0; settle-- { // auto exposure after (re)open
			cam.Capture(buf)
		}
		t0 := time.Now()
		var fr camera.Frame
		for try := 0; try < 3; try++ {
			if fr, err = cam.Capture(buf); err == nil {
				break
			}
		}
		if err != nil {
			println("webcam: error capture", err.Error())
			continue
		}
		ms := time.Since(t0).Milliseconds()
		meta := frame.Meta{Width: cfg.W, Height: cfg.H, Format: frame.JPEG, Sensor: frame.Sensor(cam.Sensor().Model)}
		if err := enc.WriteImage(meta, uint32(t0.UnixMicro()), fr.Data); err != nil {
			println("webcam: error write", err.Error())
			continue
		}
		println("webcam: sent", enc.Seq()-1, len(fr.Data), ms)
	}
}
