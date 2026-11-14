package ov3660

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

func initQVGA(t *testing.T) (*Sensor, *regBus) {
	t.Helper()
	b := newBus()
	s := &Sensor{Bus: b}
	if err := s.Init(mode(320, 240, sensor.YUV422)); err != nil {
		t.Fatal(err)
	}
	return s, b
}

func TestControlsNeedInit(t *testing.T) {
	s := &Sensor{Bus: newBus()}
	if err := s.SetControls(sensor.Controls{}); !errors.Is(err, ErrNotInit) {
		t.Errorf("SetControls: %v, want ErrNotInit", err)
	}
	if _, err := s.ReadControls(); !errors.Is(err, ErrNotInit) {
		t.Errorf("ReadControls: %v, want ErrNotInit", err)
	}
}

func TestManualControlsRoundTrip(t *testing.T) {
	s, b := initQVGA(t) // line 115 µs
	b.regs[regISPCtrl01] = 0xA2
	in := sensor.Controls{ExposureUS: 40_000, Gain: 8, WB: [3]float32{1.07, 1, 1.93}}
	if err := s.SetControls(in); err != nil {
		t.Fatal(err)
	}
	if b.regs[regAECManual] != aecManual|agcManual || b.regs[regAWBManual] != 1 {
		t.Errorf("modes 0x3503 %#x 0x3406 %#x, want manual", b.regs[regAECManual], b.regs[regAWBManual])
	}
	// 40 ms / 115 µs = 347.8 → 348 lines; register = lines << 4.
	if got := uint32(b.regs[0x3500])<<16 | uint32(b.regs[0x3501])<<8 | uint32(b.regs[0x3502]); got != 348<<4 {
		t.Errorf("exposure register %#x, want %#x", got, 348<<4)
	}
	if b.get16(regGain) != 128 {
		t.Errorf("gain register %d, want 128", b.get16(regGain))
	}
	if b.regs[regISPCtrl01] != 0xA3 {
		t.Errorf("ISP_CONTROL_01 %#x, want AWB block on, other bits kept (0xA3)", b.regs[regISPCtrl01])
	}
	out, err := s.ReadControls()
	if err != nil {
		t.Fatal(err)
	}
	if out.AutoExposure || out.AutoGain || out.AutoWB {
		t.Errorf("read back auto flags %v %v %v, want all manual", out.AutoExposure, out.AutoGain, out.AutoWB)
	}
	if out.ExposureUS != 348*115 || out.Gain != 8 {
		t.Errorf("read back %d µs ×%v, want %d µs ×8", out.ExposureUS, out.Gain, 348*115)
	}
	for i, g := range in.WB {
		if math.Abs(float64(out.WB[i]-g)) > 1.0/wbUnity {
			t.Errorf("WB[%d] = %v, want %v ± 1/1024", i, out.WB[i], g)
		}
	}
}

func TestAutoControlsReadLoopValues(t *testing.T) {
	s, b := initQVGA(t)
	if err := s.SetControls(sensor.Controls{AutoExposure: true, AutoGain: true, AutoWB: true}); err != nil {
		t.Fatal(err)
	}
	if b.regs[regAECManual] != 0 || b.regs[regAWBManual] != 0 {
		t.Errorf("modes 0x3503 %#x 0x3406 %#x, want auto", b.regs[regAECManual], b.regs[regAWBManual])
	}
	// The auto loops' values (C1: 90.2 ms at 115 µs lines = 784 lines, ×15.5, AWB 1.07/1.00/1.93).
	b.regs[0x3500], b.regs[0x3501], b.regs[0x3502] = 0x00, 0x31, 0x00
	b.regs[0x350A], b.regs[0x350B] = 0x00, 248
	for i, v := range []uint16{0x447, 0x400, 0x7B8} {
		b.regs[regAWBLive+uint16(2*i)], b.regs[regAWBLive+uint16(2*i)+1] = uint8(v>>8), uint8(v)
	}
	b.regs[regWBGain] = 0x0F // stale manual gains must not be reported in auto
	c, err := s.ReadControls()
	if err != nil {
		t.Fatal(err)
	}
	if !c.AutoExposure || !c.AutoGain || !c.AutoWB {
		t.Errorf("auto flags %v %v %v", c.AutoExposure, c.AutoGain, c.AutoWB)
	}
	if c.ExposureUS != 784*115 || c.Gain != 15.5 {
		t.Errorf("%d µs ×%v, want %d µs ×15.5", c.ExposureUS, c.Gain, 784*115)
	}
	if !near(c.WB[0], 0x447/1024.0) || c.WB[1] != 1 || !near(c.WB[2], 0x7B8/1024.0) {
		t.Errorf("WB %v, want the loop's 0x519F/A1/A3 gains", c.WB)
	}
}

func TestControlEncodingClamps(t *testing.T) {
	for _, tt := range []struct {
		g    float32
		want uint16
	}{{0.5, 16}, {float32(math.NaN()), 16}, {1, 16}, {1.5, 24}, {63.9, 1022}, {100, 1023}} {
		if got := gainReg(tt.g); got != tt.want {
			t.Errorf("gainReg(%v) = %d, want %d", tt.g, got, tt.want)
		}
	}
	for _, tt := range []struct {
		g    float32
		want uint16
	}{{-1, 0}, {float32(math.NaN()), 0}, {1, 0x400}, {3.9995, 0xFFF}, {5, 0xFFF}} {
		if got := wbReg(tt.g); got != tt.want {
			t.Errorf("wbReg(%v) = %#x, want %#x", tt.g, got, tt.want)
		}
	}
	if got := exposureReg(1<<31, 1000); got != maxExposure {
		t.Errorf("exposureReg overflow %#x, want clamp %#x", got, maxExposure)
	}
	if got := exposureReg(57, 115_000); got != 0 { // under half a line
		t.Errorf("exposureReg(57 µs) = %#x, want 0", got)
	}
}

// A manual exposure longer than the frame stretches the frame (VTS =
// exposure lines + vtsMargin, at most 0xFFFF): the sensor cannot expose
// longer than a frame (BADGECAM: 0.5–1 s exposures). A shorter exposure or
// auto exposure restores the mode's VTS.
func TestLongExposureStretchesFrame(t *testing.T) {
	b := newBus()
	s := &Sensor{Bus: b}
	if err := s.Init(mode(2048, 1536, sensor.JPEG)); err != nil {
		t.Fatal(err)
	}
	const modeVTS = 1564 // 4:3 entry; line 57.5 µs (HTS 2300 at 40 MHz SYSCLK)
	if s.LineNS() != 57_500 || b.get16(regXTotal+2) != modeVTS {
		t.Fatalf("line %d ns, VTS %d; want 57500, %d", s.LineNS(), b.get16(regXTotal+2), modeVTS)
	}
	exposure := func() uint32 {
		return (uint32(b.regs[regExposure])<<16 | uint32(b.regs[regExposure+1])<<8 | uint32(b.regs[regExposure+2])) >> 4
	}
	for _, tt := range []struct {
		name  string
		c     sensor.Controls
		lines uint32
		vts   uint16
	}{
		{"1 s", sensor.Controls{ExposureUS: 1_000_000, Gain: 1}, 17391, 17391 + vtsMargin},
		{"10 ms fits the mode's frame", sensor.Controls{ExposureUS: 10_000, Gain: 1}, 174, modeVTS},
		{"exactly the frame", sensor.Controls{ExposureUS: (modeVTS - vtsMargin) * 57_500 / 1000, Gain: 1}, modeVTS - vtsMargin, modeVTS},
		{"one line more", sensor.Controls{ExposureUS: (modeVTS - vtsMargin + 1) * 57_500 / 1000, Gain: 1}, modeVTS - vtsMargin + 1, modeVTS + 1},
		{"10 s clamps to the longest frame", sensor.Controls{ExposureUS: 10_000_000, Gain: 1}, 0xFFFF - vtsMargin, 0xFFFF},
	} {
		if err := s.SetControls(tt.c); err != nil {
			t.Fatal(err)
		}
		if got := exposure(); got != tt.lines || b.get16(regXTotal+2) != tt.vts {
			t.Errorf("%s: %d lines, VTS %d; want %d, %d", tt.name, got, b.get16(regXTotal+2), tt.lines, tt.vts)
		}
	}
	if err := s.SetControls(sensor.Controls{ExposureUS: 1_000_000, Gain: 1}); err != nil {
		t.Fatal(err)
	}
	if err := s.SetControls(sensor.Controls{AutoExposure: true, AutoGain: true, AutoWB: true}); err != nil {
		t.Fatal(err)
	}
	if b.get16(regXTotal+2) != modeVTS {
		t.Errorf("auto exposure after 1 s: VTS %d, want the mode's %d", b.get16(regXTotal+2), modeVTS)
	}
}

// Mirror and flip use the sensor readout bit (bit 1 of TIMING_TC_REG21 /
// REG20) only: bit 2 (the ISP mirror/flip, which esp32-camera also sets)
// shifts the Bayer phase and turns the whole image magenta (BADGECAM:
// G/R 0.51 with both bits, 1.14 with bit 1 alone, 1.15 unmirrored).
// SetMirror changes only those bits, and the choice survives a later Init
// (a new size), binned or not. The OV3660 on the XIAO Sense delivers a
// left-right mirrored image without hmirror (the badge's silkscreen reads
// backwards).
func TestMirrorFlip(t *testing.T) {
	const bits = 0x02
	b := newBus()
	s := &Sensor{Bus: b}
	if err := s.Init(mode(2048, 1536, sensor.JPEG)); err != nil {
		t.Fatal(err)
	}
	r20, r21 := b.regs[regTiming20], b.regs[regTiming21]
	if r20&bits != 0 || r21&bits != 0 {
		t.Fatalf("Init without SetMirror: 0x3820 %#x 0x3821 %#x, want bits 2:1 clear", r20, r21)
	}
	if err := s.SetMirror(true, false); err != nil {
		t.Fatal(err)
	}
	if b.regs[regTiming21] != r21|bits || b.regs[regTiming20] != r20 {
		t.Errorf("hmirror: 0x3820 %#x 0x3821 %#x, want %#x %#x", b.regs[regTiming20], b.regs[regTiming21], r20, r21|bits)
	}
	if err := s.SetMirror(false, true); err != nil {
		t.Fatal(err)
	}
	if b.regs[regTiming21] != r21 || b.regs[regTiming20] != r20|bits {
		t.Errorf("vflip: 0x3820 %#x 0x3821 %#x, want %#x %#x", b.regs[regTiming20], b.regs[regTiming21], r20|bits, r21)
	}
	if err := s.SetMirror(true, true); err != nil {
		t.Fatal(err)
	}
	if err := s.Init(mode(640, 480, sensor.JPEG)); err != nil { // binned: other bits differ
		t.Fatal(err)
	}
	if b.regs[regTiming20] != 0x01|bits || b.regs[regTiming21] != 0x21|bits {
		t.Errorf("after Init (binned JPEG): 0x3820 %#x 0x3821 %#x, want %#x %#x", b.regs[regTiming20], b.regs[regTiming21], 0x01|bits, 0x21|bits)
	}
}
