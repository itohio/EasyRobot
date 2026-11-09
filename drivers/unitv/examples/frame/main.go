//go:build unitv

// Command frame streams camera images over UARTHS in the camera frame format
// v1 (drivers/camera/frame), for cmd/camframe on the host (milestone M4).
//
// Build-time options (tinygo -ldflags "-X main.mode=rgb565-qvga -X main.resend=1"):
// mode is y8-qqvga, y8-qvga, rgb565-qqvga or rgb565-qvga; baud is the UART
// rate (default 1500000); burst/gap pace the output (idle gap ms after every
// burst bytes; burst 0 = unpaced); crop=1 takes a centred window instead of
// the full field; pattern=1 the sensor's colour bars; resend=1 keeps each
// image until the host's Ack and resends Nak'ed chunks (camframe -resend).
package main

import (
	"io"
	"machine"
	"strconv"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/unitv"
)

var (
	mode    = "y8-qqvga"
	baud    = "1500000"
	burst   = "256"
	gap     = "2"
	crop    = "0" // 1: centred window at 1:1 sensor pixels; 0: full field scaled
	pattern = "0" // 1: sensor colour bars
	resend  = "0" // 1: keep each image until the host's Ack, resend Nak'ed chunks
)

// awaitAck answers the host's Naks for image seq with the chunk again until
// its Ack, or until the host has been silent for a second.
func awaitAck(uart *machine.UART, enc *frame.Encoder, p *frame.ControlParser, seq uint32) {
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		for uart.Buffered() > 0 {
			b, _ := uart.ReadByte()
			c, ok := p.Feed(b)
			if !ok || c.Seq != seq {
				continue
			}
			if c.Kind == frame.Ack {
				return
			}
			if err := enc.Resend(int(c.Chunk)); err != nil {
				println("frame: resend:", err.Error())
			}
			deadline = time.Now().Add(time.Second)
		}
	}
	println("frame: no ack for", seq)
}

func config() (camera.Config, frame.Format) {
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.Y8}}
	f := frame.Y8
	switch mode {
	case "y8-qvga":
		cfg.Resolution = sensor.QVGA
	case "rgb565-qqvga":
		cfg.Format, f = sensor.RGB565, frame.RGB565LE
	case "rgb565-qvga":
		cfg.Resolution, cfg.Format, f = sensor.QVGA, sensor.RGB565, frame.RGB565LE
	}
	cfg.Crop = crop == "1"
	cfg.TestPattern = pattern == "1"
	return cfg, f
}

// align8 returns an 8-byte aligned window of n bytes (the DVP writes with
// 64-bit AXI bursts).
func align8(n int) []byte {
	b := make([]byte, n+8)
	off := int(-uintptr(unsafe.Pointer(&b[0])) & 7)
	return b[off : off+n]
}

// paced writes to w and idles gap after every burst bytes, so UnitV's FTDI
// FT232BM (384-byte buffer) and the host can drain: unpaced 1.5 Mbaud loses
// bytes in ~6% of chunks, 256 B / 2 ms in ~1% (ER-K210-M4 evidence, outside the repo).
type paced struct {
	w     *machine.UART
	burst int
	gap   time.Duration
	n     int
}

func (p *paced) Write(b []byte) (int, error) {
	total := len(b)
	for len(b) > 0 {
		k := min(len(b), p.burst-p.n)
		p.w.Write(b[:k])
		b, p.n = b[k:], p.n+k
		if p.n == p.burst {
			p.n = 0
			end := time.Now().Add(p.gap)
			for time.Now().Before(end) {
			}
		}
	}
	return total, nil
}

func main() {
	rate, _ := strconv.Atoi(baud)
	uart := machine.UART0
	uart.Configure(machine.UARTConfig{BaudRate: uint32(rate)})
	println("frame: mode", mode, "baud", rate)

	cfg, f := config()
	cam, err := camera.Open(cfg, unitv.CameraPins)
	for err != nil {
		println("frame: open:", err.Error())
		time.Sleep(time.Second)
		cam, err = camera.Open(cfg, unitv.CameraPins)
	}
	id := cam.Sensor()
	println("frame: sensor", id.Model.String(), "xclk", camera.ClockInfo().XCLK)

	buf := align8(camera.FrameSize(cfg))
	burstN, _ := strconv.Atoi(burst)
	gapMs, _ := strconv.Atoi(gap)
	var out io.Writer = uart
	if burstN > 0 {
		out = &paced{w: uart, burst: burstN, gap: time.Duration(gapMs) * time.Millisecond}
	}
	enc := frame.NewEncoder(out)
	meta := frame.Meta{Width: cfg.W, Height: cfg.H, Format: f, Sensor: frame.Sensor(id.Model)}

	for range 10 { // let auto exposure settle
		if _, err := cam.Capture(buf); err != nil {
			println("frame: warm-up capture:", err.Error())
		}
	}
	var parser frame.ControlParser
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
			continue
		}
		if resend == "1" {
			awaitAck(uart, enc, &parser, enc.Seq()-1)
		}
	}
}
