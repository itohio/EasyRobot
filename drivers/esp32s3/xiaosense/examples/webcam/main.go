//go:build xiao_esp32s3

// Command webcam turns the XIAO ESP32-S3 Sense into a USB webcam: OV3660
// JPEGs up to 2048×1536, captured by GDMA straight into a PSRAM buffer and
// sent over the USB serial port in the camera frame format v1
// (drivers/camera/frame), one on request or back to back while streaming,
// with manual or auto exposure and gain. No radio.
//
// Commands are text lines (see cmd.go): G (one image), S/X (start/stop
// streaming), Z (size), Q (quality), E (exposure), I (gain), W (white
// balance), M (mirror and flip), ? (status).
// Replies are frame-v1 images and text lines starting "webcam: ":
//
//	webcam: status size=0 w=2048 h=1536 q=30 ae=1 exp_us=33120 ag=0 gain=1.00 awb=1 wb=1.47,1.00,1.31 hm=1 vf=0 stream=1 fps=3.1 bytes=231916 fails=0
//	webcam: error <what>
//
// While streaming, a status line follows every 2 s and every command. At
// boot: size 0, quality 30, auto exposure, manual gain 1 (the lowest, least
// noise; the badge-cam default), auto white balance, left-right mirror on
// (this module's OV3660 image is mirrored without it, so text reads
// backwards). Streaming stops by
// itself when the host stops reading (3 writes in a row time out).
//
// Hosts: cmd/camclient (live view and controls in a browser), or
// cmd/camframe -trigger $'G 0 30\n' for single images.
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

// lineReader collects command lines from the USB serial port without
// blocking.
type lineReader struct {
	buf [48]byte
	n   int
}

// poll returns the next complete line, if one has arrived.
func (r *lineReader) poll() ([]byte, bool) {
	for machine.Serial.Buffered() > 0 {
		b, err := machine.Serial.ReadByte()
		if err != nil {
			return nil, false
		}
		if b == '\n' || b == '\r' {
			if r.n == 0 {
				continue
			}
			l := r.buf[:r.n]
			r.n = 0
			return l, true
		}
		if r.n < len(r.buf) {
			r.buf[r.n] = b
			r.n++
		}
	}
	return nil, false
}

type webcam struct {
	cam     *camera.Camera
	buf     []byte
	enc     *frame.Encoder
	size    int
	quality uint8
	ctl     sensor.Controls
	mirrorH bool
	mirrorV bool
	stream  bool
	wfails  int     // consecutive failed image writes while streaming
	fps     float32 // smoothed images sent per second while streaming
	last    time.Time
	bytes   int
	capFail uint32 // captures that failed (after 3 tries) since boot
}

func fatal(what string, err error) {
	for {
		println("webcam: error", what, err.Error())
		time.Sleep(time.Second)
	}
}

// open (re)opens the camera at w.size and applies quality and controls.
// The first frames after it let auto exposure settle.
func (w *webcam) open() bool {
	if w.cam != nil {
		w.cam.Close()
		w.cam = nil
	}
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.Resolution{W: sizeW[w.size], H: sizeH[w.size]}, Format: sensor.JPEG}}
	cam, err := camera.Open(cfg, xiaosense.CameraPins)
	if err != nil {
		println("webcam: error open", err.Error())
		return false
	}
	cam.SetCacheSync(psram.Cache{})
	w.cam = cam
	w.setQuality()
	w.setMirror()
	w.setControls()
	settle := 8
	if !w.ctl.AutoExposure {
		settle = 2
	}
	for range settle {
		cam.Capture(w.buf)
	}
	return true
}

func (w *webcam) setQuality() {
	if d, ok := w.cam.Driver().(*ov3660.Sensor); ok {
		if err := d.SetQuality(w.quality); err != nil {
			println("webcam: error quality", err.Error())
		}
	}
}

func (w *webcam) setMirror() {
	if d, ok := w.cam.Driver().(*ov3660.Sensor); ok {
		if err := d.SetMirror(w.mirrorH, w.mirrorV); err != nil {
			println("webcam: error mirror", err.Error())
		}
	}
}

func (w *webcam) setControls() {
	if err := w.cam.SetControls(w.ctl); err != nil {
		println("webcam: error controls", err.Error())
	}
}

func f2(v float32) string { return strconv.FormatFloat(float64(v), 'f', 2, 32) }

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// status prints the status line with the sensor's live exposure and gain.
func (w *webcam) status() {
	ctl := w.ctl
	if w.cam != nil {
		if c, err := w.cam.ReadControls(); err == nil {
			ctl.ExposureUS, ctl.Gain, ctl.WB = c.ExposureUS, c.Gain, c.WB
		}
	}
	println("webcam: status size="+strconv.Itoa(w.size), "w="+strconv.Itoa(int(sizeW[w.size])), "h="+strconv.Itoa(int(sizeH[w.size])),
		"q="+strconv.Itoa(int(w.quality)), "ae="+strconv.Itoa(b2i(w.ctl.AutoExposure)), "exp_us="+strconv.Itoa(int(ctl.ExposureUS)),
		"ag="+strconv.Itoa(b2i(w.ctl.AutoGain)), "gain="+strconv.FormatFloat(float64(ctl.Gain), 'f', 2, 32),
		"awb="+strconv.Itoa(b2i(w.ctl.AutoWB)), "wb="+f2(ctl.WB[0])+","+f2(ctl.WB[1])+","+f2(ctl.WB[2]),
		"hm="+strconv.Itoa(b2i(w.mirrorH)), "vf="+strconv.Itoa(b2i(w.mirrorV)), "stream="+strconv.Itoa(b2i(w.stream)), "fps="+strconv.FormatFloat(float64(w.fps), 'f', 1, 32), "bytes="+strconv.Itoa(w.bytes), "fails="+strconv.Itoa(int(w.capFail)))
}

// send captures one image and writes it; it reports whether the write
// succeeded (a capture error is reported and counts as success: the host is
// still there).
func (w *webcam) send() bool {
	if w.cam == nil && !w.open() {
		time.Sleep(time.Second)
		return true
	}
	var fr camera.Frame
	var err error
	for range 3 {
		if fr, err = w.cam.Capture(w.buf); err == nil {
			break
		}
	}
	if err != nil {
		w.capFail++
		println("webcam: error capture", err.Error())
		return true
	}
	t := time.Now()
	meta := frame.Meta{Width: sizeW[w.size], Height: sizeH[w.size], Format: frame.JPEG, Sensor: frame.Sensor(w.cam.Sensor().Model)}
	if err := w.enc.WriteImage(meta, uint32(t.UnixMicro()), fr.Data); err != nil {
		return false
	}
	w.bytes = len(fr.Data)
	if dt := time.Since(w.last).Seconds(); !w.last.IsZero() && dt > 0 {
		w.fps += (1/float32(dt) - w.fps) / 4
	}
	w.last = t
	return true
}

func (w *webcam) handle(c command) {
	switch c.kind {
	case cmdNone:
		return
	case cmdBad:
		println("webcam: error command")
		return
	case cmdGrab:
		if c.size != w.size || w.cam == nil {
			w.size = c.size
			if !w.open() {
				return
			}
		}
		if c.quality != w.quality {
			w.quality = c.quality
			w.setQuality()
		}
		if w.send() {
			println("webcam: sent", w.enc.Seq()-1, w.bytes)
		}
		return
	case cmdStart:
		w.stream, w.wfails, w.fps, w.last = true, 0, 0, time.Time{}
		return
	case cmdStop:
		w.stream = false
	case cmdSize:
		w.size = c.size
		w.fps, w.last = 0, time.Time{}
		w.open()
	case cmdQuality:
		w.quality = c.quality
		if w.cam != nil {
			w.setQuality()
		}
	case cmdExposure:
		w.ctl.AutoExposure = c.auto
		if !c.auto {
			w.ctl.ExposureUS = c.exposure
		}
		w.fps, w.last = 0, time.Time{}
		if w.cam != nil {
			w.setControls()
		}
	case cmdMirror:
		w.mirrorH, w.mirrorV = c.h, c.v
		if w.cam != nil {
			w.setMirror()
		}
	case cmdWB:
		w.ctl.AutoWB = c.auto
		if !c.auto {
			w.ctl.WB = c.wb
		}
		if w.cam != nil {
			w.setControls()
		}
	case cmdGain:
		w.ctl.AutoGain = c.auto
		if !c.auto {
			w.ctl.Gain = c.gain
		}
		if w.cam != nil {
			w.setControls()
		}
	}
	w.status()
}

func main() {
	time.Sleep(time.Second)
	r, _, err := psram.Init()
	if err != nil {
		fatal("psram", err)
	}
	arena := psram.NewArena(r)
	buf, err := arena.Alloc(camera.JPEGBufSize(camera.Config{Mode: sensor.Mode{Resolution: sensor.Resolution{W: sizeW[0], H: sizeH[0]}, Format: sensor.JPEG}}))
	if err != nil {
		fatal("alloc", err)
	}
	w := &webcam{buf: buf, enc: frame.NewEncoder(usbWriter{}), quality: defaultQuality,
		ctl: sensor.Controls{AutoExposure: true, AutoGain: false, Gain: minGain, AutoWB: true}, mirrorH: true}
	println("webcam: ready")
	var lr lineReader
	lastStatus := time.Now()
	for {
		if l, ok := lr.poll(); ok {
			w.handle(parseCommand(l))
			continue
		}
		if !w.stream {
			time.Sleep(2 * time.Millisecond)
			continue
		}
		if w.send() {
			w.wfails = 0
		} else if w.wfails++; w.wfails >= 3 {
			w.stream = false // host stopped reading
			continue
		}
		if time.Since(lastStatus) >= 2*time.Second {
			w.status()
			lastStatus = time.Now()
		}
	}
}
