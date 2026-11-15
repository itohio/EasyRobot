//go:build unitv

// Command controls exercises the camera's exposure/gain/white-balance
// controls: auto first, then a manual exposure sweep at the
// auto gain (1 line as black reference, then E_auto × ½, 1, 2, 4), printing
// read-backs and flat-patch statistics and sending one QQVGA Y8 frame per
// setting in frame format v1, at 115200.
//
// Build-time options: -X main.patch=x0,y0,x1,y1 (flat patch, default the wall
// at the top right of the current view).
package main

import (
	"machine"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/unitv"
)

var patch = "135,8,155,32"

func align8(n int) []byte {
	b := make([]byte, n+8)
	off := int(-uintptr(unsafe.Pointer(&b[0])) & 7)
	return b[off : off+n]
}

type stats struct{ mean, sigma, frame float32 }

func measure(px []byte, w int, x0, y0, x1, y1 int) stats {
	var sum, sq float32
	n := 0
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			v := float32(px[y*w+x])
			sum += v
			sq += v * v
			n++
		}
	}
	m := sum / float32(n)
	var all float32
	for _, v := range px {
		all += float32(v)
	}
	return stats{m, sqrt(sq/float32(n) - m*m), all / float32(len(px))}
}

func sqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}
	r := x
	for range 20 {
		r = (r + x/r) / 2
	}
	return r
}

func f1(v float32) string { return strconv.FormatFloat(float64(v), 'f', 2, 32) }

func printControls(tag string, c sensor.Controls) {
	println(tag, "auto_exp", c.AutoExposure, "exp_us", c.ExposureUS, "auto_gain", c.AutoGain, "gain", f1(c.Gain),
		"auto_wb", c.AutoWB, "wb", f1(c.WB[0]), f1(c.WB[1]), f1(c.WB[2]))
}

func main() {
	machine.UART0.Configure(machine.UARTConfig{BaudRate: 115200})
	var p [4]int
	for i, s := range strings.Split(patch, ",") {
		p[i], _ = strconv.Atoi(s)
	}
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.Y8}}
	cam, err := camera.Open(cfg, unitv.CameraPins)
	if err != nil {
		println("controls: open:", err.Error())
		return
	}
	buf := align8(camera.FrameSize(cfg))
	enc := frame.NewEncoder(machine.UART0)
	meta := frame.Meta{Width: cfg.W, Height: cfg.H, Format: frame.Y8, Sensor: frame.SensorOV2640}

	capture := func(tag string, settle int) stats {
		var fr camera.Frame
		var t [2]time.Time
		for i := range settle + 2 {
			if i == settle {
				t[0] = time.Now()
			}
			fr, err = cam.Capture(buf)
			if err != nil {
				println("controls: capture:", err.Error())
				return stats{}
			}
			if i == settle {
				t[1] = time.Now()
			}
		}
		st := measure(fr.Data, int(fr.Width), p[0], p[1], p[2], p[3])
		println(tag, "patch_mean", f1(st.mean), "patch_sigma", f1(st.sigma), "frame_mean", f1(st.frame),
			"frame_period_us", t[1].Sub(t[0]).Microseconds())
		enc.WriteImage(meta, 0, fr.Data)
		println()
		return st
	}

	for range 10 { // let the auto loops settle
		cam.Capture(buf)
	}
	auto, err := cam.ReadControls()
	if err != nil {
		println("controls: read:", err.Error())
		return
	}
	printControls("C1 auto readback:", auto)
	sa := capture("C1 auto:", 0)

	// Sweep at the auto gain: 1 line (black reference), then E_auto × ½, 1, 2, 4.
	ea := auto.ExposureUS
	exps := [5]uint32{1, ea / 2, ea, 2 * ea, 4 * ea}
	var res [5]stats
	for i, e := range exps {
		set := sensor.Controls{ExposureUS: e, Gain: auto.Gain, AutoWB: true}
		if err := cam.SetControls(set); err != nil {
			println("controls: set:", err.Error())
			return
		}
		got, _ := cam.ReadControls()
		printControls("C1 manual set exp_us "+strconv.Itoa(int(e))+" readback:", got)
		res[i] = capture("C1 manual exp_us "+strconv.Itoa(int(e))+":", 3)
	}
	b := res[0].mean
	println("C1 black", f1(b), "ratio (m(2E)-b)/(m(E)-b) at E=E_auto:", f1((res[3].mean-b)/(res[2].mean-b)),
		"at E=E_auto/2:", f1((res[2].mean-b)/(res[1].mean-b)), "raw m(2E)/m(E) at E_auto:", f1(res[3].mean/res[2].mean),
		"auto patch_mean", f1(sa.mean))
	for {
		time.Sleep(time.Second)
	}
}
