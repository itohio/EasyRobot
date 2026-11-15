package ov2640

import (
	"errors"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// recBus keeps the last value written to each (bank, register) and the order
// of writes, following BANK_SEL like the chip does.
type recBus struct {
	bank   uint8
	last   map[[2]uint8]uint8
	writes [][3]uint8 // bank, reg, val
}

func (b *recBus) WriteReg8(dev, reg, val uint8) error {
	if dev != Addr {
		return errors.New("wrong device")
	}
	if reg == bankSel {
		b.bank = val
		return nil
	}
	if b.last == nil {
		b.last = map[[2]uint8]uint8{}
	}
	b.last[[2]uint8{b.bank, reg}] = val
	b.writes = append(b.writes, [3]uint8{b.bank, reg, val})
	return nil
}
func (b *recBus) ReadReg8(dev, reg uint8) (uint8, error)            { return 0, nil }
func (b *recBus) ReadReg16(dev uint8, reg uint16) (uint8, error)    { return 0, nil }
func (b *recBus) WriteReg16(dev uint8, reg uint16, val uint8) error { return nil }

func TestInitProgramsOutputSizeAndFormat(t *testing.T) {
	tests := []struct {
		mode      sensor.Mode
		lowFirst  bool
		imageMode uint8
		zmow      uint8
		zmoh      uint8
	}{
		{sensor.Mode{Resolution: sensor.QVGA, Format: sensor.RGB565}, false, imageModeRGB565, 80, 60},
		{sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.Y8}, false, imageModeYUV422, 40, 30},
		{sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.Y8}, true, imageModeYUV422 | imageModeLByteFirst, 40, 30},
		{sensor.Mode{Resolution: sensor.QVGA, Format: sensor.YUV422}, false, imageModeYUV422, 80, 60},
		{sensor.Mode{Resolution: sensor.QVGA, Format: sensor.RGB565}, true, imageModeRGB565, 80, 60}, // option is YUV-only
		{sensor.Mode{Resolution: sensor.QVGA, Format: sensor.JPEG}, false, imageModeJPEG | imageModeHV, 80, 60},
		{sensor.Mode{Resolution: sensor.Resolution{W: 256, H: 256}, Format: sensor.RGB888Planar}, false, imageModeRGB565, 64, 64}, // RGB565 on the bus; the capture side splits planes
	}
	for _, tt := range tests {
		bus := &recBus{}
		s := &Sensor{Bus: bus, Opts: Options{ClockDouble: true, ClockDiv: 3, YUVLowByteFirst: tt.lowFirst}}
		if err := s.Init(tt.mode); err != nil {
			t.Fatalf("%v: %v", tt.mode, err)
		}
		if got := bus.writes[0]; got != [3]uint8{bankSensor, com7, com7SRST} {
			t.Errorf("%v: first write %v, want sensor soft reset", tt.mode, got)
		}
		dsp := func(r uint8) uint8 { return bus.last[[2]uint8{bankDSP, r}] }
		if dsp(zmow) != tt.zmow || dsp(zmoh) != tt.zmoh || dsp(zmhh) != 0 {
			t.Errorf("%v: output size regs %d×%d (zmhh %d), want %d×%d", tt.mode, dsp(zmow), dsp(zmoh), dsp(zmhh), tt.zmow, tt.zmoh)
		}
		if dsp(imageMode) != tt.imageMode {
			t.Errorf("%v: IMAGE_MODE %#x, want %#x", tt.mode, dsp(imageMode), tt.imageMode)
		}
		if dsp(rBypass) != 0 {
			t.Errorf("%v: DSP left bypassed", tt.mode)
		}
		wantClk := uint8(0x83)
		if tt.mode.Format == sensor.JPEG {
			wantClk = 0
		}
		if got := bus.last[[2]uint8{bankSensor, clkrc}]; got != wantClk {
			t.Errorf("%v: CLKRC %#x, want %#x", tt.mode, got, wantClk)
		}
		if got := bus.last[[2]uint8{bankSensor, com7}]; got != com7ResCIF {
			t.Errorf("%v: COM7 %#x, want CIF without colour bars", tt.mode, got)
		}
	}
}

func TestInitTestPatternSetsColourBarLast(t *testing.T) {
	bus := &recBus{}
	m := sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.RGB565, TestPattern: true}
	if err := (&Sensor{Bus: bus}).Init(m); err != nil {
		t.Fatal(err)
	}
	last := bus.writes[len(bus.writes)-1]
	if last != [3]uint8{bankSensor, com7, com7ResCIF | com7ColorBar} {
		t.Fatalf("last write %v, want COM7 = CIF|colour bar in the sensor bank", last)
	}
}

func TestInitWindowFullFieldVersusCrop(t *testing.T) {
	tests := []struct {
		crop                     bool
		hsize, vsize, xoff, yoff uint8
	}{
		{false, 100, 74, 0, 0},  // whole 400×296 CIF frame, scaled to 160×120
		{true, 40, 30, 120, 88}, // centred 160×120 at 1:1
	}
	for _, tt := range tests {
		bus := &recBus{}
		m := sensor.Mode{Resolution: sensor.QQVGA, Format: sensor.Y8, Crop: tt.crop}
		if err := (&Sensor{Bus: bus}).Init(m); err != nil {
			t.Fatal(err)
		}
		dsp := func(r uint8) uint8 { return bus.last[[2]uint8{bankDSP, r}] }
		if dsp(hSize) != tt.hsize || dsp(vSize) != tt.vsize || dsp(xOffL) != tt.xoff || dsp(yOffL) != tt.yoff || dsp(vhyx) != 0 {
			t.Errorf("crop=%v: window %d×%d at (%d,%d) vhyx %#x, want %d×%d at (%d,%d) vhyx 0",
				tt.crop, dsp(hSize), dsp(vSize), dsp(xOffL), dsp(yOffL), dsp(vhyx), tt.hsize, tt.vsize, tt.xoff, tt.yoff)
		}
		if dsp(zmow) != 40 || dsp(zmoh) != 30 {
			t.Errorf("crop=%v: output %d×%d units, want 40×30", tt.crop, dsp(zmow), dsp(zmoh))
		}
	}
}

func TestInitRejectsUnsupportedModes(t *testing.T) {
	for _, m := range []sensor.Mode{
		{Resolution: sensor.Resolution{W: 1604, H: 1200}, Format: sensor.JPEG}, // wider than UXGA
		{Resolution: sensor.Resolution{W: 1600, H: 1204}, Format: sensor.JPEG}, // taller than UXGA
		{Resolution: sensor.Resolution{W: 322, H: 240}, Format: sensor.RGB565}, // not a multiple of 4
		{Resolution: sensor.QVGA, Format: 0},
	} {
		bus := &recBus{}
		if err := (&Sensor{Bus: bus}).Init(m); !errors.Is(err, ErrMode) {
			t.Errorf("%v: err = %v, want ErrMode", m, err)
		}
		if len(bus.writes) != 0 || bus.bank != 0 {
			t.Errorf("%v: %d register writes before rejecting the mode", m, len(bus.writes))
		}
	}
}

// TestInitSensorModes: the smallest of CIF, SVGA and UXGA that holds the
// output is chosen, with its whole field as the DSP window and the
// esp32-camera clocks (raw: CLKRC ÷7 above CIF, auto PCLK ÷8, ÷12 in UXGA;
// JPEG: CLKRC 0, manual PCLK ÷8, ÷12 in UXGA).
func TestInitSensorModes(t *testing.T) {
	tests := []struct {
		mode               sensor.Mode
		com7               uint8
		hsize, vsize, vhyx uint8
		zmow, zmoh, zmhh   uint8
		clkrc, dvpsp       uint8
	}{
		{sensor.Mode{Resolution: sensor.Resolution{W: 400, H: 296}, Format: sensor.RGB565}, com7ResCIF, 100, 74, 0, 100, 74, 0, 0x83, 0x88},
		{sensor.Mode{Resolution: sensor.VGA, Format: sensor.RGB565}, com7ResSVGA, 200, 150, 0, 160, 120, 0, 0x87, 0x88},
		{sensor.Mode{Resolution: sensor.Resolution{W: 800, H: 600}, Format: sensor.JPEG}, com7ResSVGA, 200, 150, 0, 200, 150, 0, 0, 8},
		{sensor.Mode{Resolution: sensor.Resolution{W: 804, H: 600}, Format: sensor.JPEG}, com7ResUXGA, 0x90, 0x2c, 0x88, 201, 150, 0, 0, 12},
		{sensor.Mode{Resolution: sensor.Resolution{W: 1600, H: 1200}, Format: sensor.JPEG}, com7ResUXGA, 0x90, 0x2c, 0x88, 0x90, 0x2c, 0x05, 0, 12},
		{sensor.Mode{Resolution: sensor.Resolution{W: 1600, H: 1200}, Format: sensor.RGB565}, com7ResUXGA, 0x90, 0x2c, 0x88, 0x90, 0x2c, 0x05, 0x87, 0x8c},
	}
	for _, tt := range tests {
		bus := &recBus{}
		s := &Sensor{Bus: bus, Opts: Options{ClockDouble: true, ClockDiv: 3}}
		if err := s.Init(tt.mode); err != nil {
			t.Fatalf("%v: %v", tt.mode, err)
		}
		dsp := func(r uint8) uint8 { return bus.last[[2]uint8{bankDSP, r}] }
		sen := func(r uint8) uint8 { return bus.last[[2]uint8{bankSensor, r}] }
		if sen(com7) != tt.com7 {
			t.Errorf("%v: COM7 %#x, want %#x", tt.mode, sen(com7), tt.com7)
		}
		if dsp(hSize) != tt.hsize || dsp(vSize) != tt.vsize || dsp(vhyx) != tt.vhyx || dsp(test) != 0 {
			t.Errorf("%v: window %#x×%#x vhyx %#x test %#x, want %#x×%#x vhyx %#x", tt.mode, dsp(hSize), dsp(vSize), dsp(vhyx), dsp(test), tt.hsize, tt.vsize, tt.vhyx)
		}
		if dsp(zmow) != tt.zmow || dsp(zmoh) != tt.zmoh || dsp(zmhh) != tt.zmhh {
			t.Errorf("%v: output %#x×%#x zmhh %#x, want %#x×%#x zmhh %#x", tt.mode, dsp(zmow), dsp(zmoh), dsp(zmhh), tt.zmow, tt.zmoh, tt.zmhh)
		}
		if sen(clkrc) != tt.clkrc || dsp(rDVPSP) != tt.dvpsp {
			t.Errorf("%v: CLKRC %#x R_DVP_SP %#x, want %#x %#x", tt.mode, sen(clkrc), dsp(rDVPSP), tt.clkrc, tt.dvpsp)
		}
	}
}

// TestJPEGSensorHREF: the option clears IMAGE_MODE's HREF = VSYNC bit;
// without it the esp32-camera setting stays.
func TestJPEGSensorHREF(t *testing.T) {
	for _, opt := range []bool{false, true} {
		bus := &recBus{}
		s := &Sensor{Bus: bus, Opts: Options{JPEGSensorHREF: opt}}
		if err := s.Init(sensor.Mode{Resolution: sensor.QVGA, Format: sensor.JPEG}); err != nil {
			t.Fatal(err)
		}
		want := uint8(imageModeJPEG | imageModeHV)
		if opt {
			want = imageModeJPEG
		}
		if got := bus.last[[2]uint8{bankDSP, imageMode}]; got != want {
			t.Errorf("JPEGSensorHREF %v: IMAGE_MODE %#x, want %#x", opt, got, want)
		}
		if got := bus.last[[2]uint8{bankDSP, reset}]; got != 0 {
			t.Errorf("JPEGSensorHREF %v: DSP left in reset (%#x)", opt, got)
		}
	}
}

func TestSetQuality(t *testing.T) {
	bus := &recBus{bank: bankSensor}
	if err := (&Sensor{Bus: bus}).SetQuality(12); err != nil {
		t.Fatal(err)
	}
	if got := bus.writes; len(got) != 1 || got[0] != [3]uint8{bankDSP, qs, 12} {
		t.Errorf("writes %v, want QS = 12 in the DSP bank", got)
	}
}

// TestCropInLargerModes: a cropped window is centred in the chosen mode's
// field (SVGA 800×600 for 640×480), not in CIF's.
func TestCropInLargerModes(t *testing.T) {
	bus := &recBus{}
	m := sensor.Mode{Resolution: sensor.VGA, Format: sensor.JPEG, Crop: true}
	if err := (&Sensor{Bus: bus}).Init(m); err != nil {
		t.Fatal(err)
	}
	dsp := func(r uint8) uint8 { return bus.last[[2]uint8{bankDSP, r}] }
	if dsp(hSize) != 160 || dsp(vSize) != 120 || dsp(xOffL) != 80 || dsp(yOffL) != 60 || dsp(vhyx) != 0 {
		t.Errorf("window %d×%d at (%d,%d) vhyx %#x, want 160×120 units at (80,60)", dsp(hSize), dsp(vSize), dsp(xOffL), dsp(yOffL), dsp(vhyx))
	}
}

// TestQualitySurvivesInit: Init starts with a
// soft reset, which drops QS; a quality set before is written again by a
// JPEG Init (and not by a raw one).
func TestQualitySurvivesInit(t *testing.T) {
	bus := &recBus{}
	s := &Sensor{Bus: bus}
	if err := s.SetQuality(40); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		f    sensor.Format
		want bool
	}{{sensor.JPEG, true}, {sensor.RGB565, false}} {
		bus.writes = nil
		if err := s.Init(sensor.Mode{Resolution: sensor.Resolution{W: 1600, H: 1200}, Format: c.f}); err != nil {
			t.Fatal(err)
		}
		reset, qsAfter := -1, false
		for i, w := range bus.writes {
			if w == [3]uint8{bankSensor, com7, com7SRST} {
				reset = i
			}
			if w == [3]uint8{bankDSP, qs, 40} && reset >= 0 && i > reset {
				qsAfter = true
			}
		}
		if qsAfter != c.want {
			t.Errorf("%v: QS 40 written after the reset %v, want %v", c.f, qsAfter, c.want)
		}
	}
}

// TestLargerModeDetails: 320×480
// needs SVGA (CIF is only 296 high); crop offsets of 256 and more reach
// VHYX's high bits; colour bars keep each mode's COM7; the JPEG HREF option
// leaves RGB565's IMAGE_MODE alone.
func TestLargerModeDetails(t *testing.T) {
	init := func(m sensor.Mode, o Options) *recBus {
		bus := &recBus{}
		if err := (&Sensor{Bus: bus, Opts: o}).Init(m); err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		return bus
	}
	dsp := func(b *recBus, r uint8) uint8 { return b.last[[2]uint8{bankDSP, r}] }
	sen := func(b *recBus, r uint8) uint8 { return b.last[[2]uint8{bankSensor, r}] }

	if b := init(sensor.Mode{Resolution: sensor.Resolution{W: 320, H: 480}, Format: sensor.RGB565}, Options{}); sen(b, com7) != com7ResSVGA {
		t.Errorf("320×480: COM7 %#x, want SVGA", sen(b, com7))
	}
	// Crop 640×800 in UXGA: offX 480 (VHYX bits 0–2 = 480>>8 = 1); crop
	// 1024×600: offY 300 (bits 4–6 = 300>>4&0x70 = 0x10).
	b := init(sensor.Mode{Resolution: sensor.Resolution{W: 640, H: 800}, Format: sensor.JPEG, Crop: true}, Options{})
	if dsp(b, xOffL) != 480&0xFF || dsp(b, vhyx)&0x07 != 1 {
		t.Errorf("crop 640×800: XOFFL %d VHYX %#x, want offX 480", dsp(b, xOffL), dsp(b, vhyx))
	}
	b = init(sensor.Mode{Resolution: sensor.Resolution{W: 1024, H: 600}, Format: sensor.JPEG, Crop: true}, Options{})
	if dsp(b, yOffL) != 300&0xFF || dsp(b, vhyx)&0x70 != 0x10 {
		t.Errorf("crop 1024×600: YOFFL %d VHYX %#x, want offY 300", dsp(b, yOffL), dsp(b, vhyx))
	}
	for _, c := range []struct {
		res  sensor.Resolution
		com7 uint8
	}{{sensor.Resolution{W: 800, H: 600}, com7ResSVGA}, {sensor.Resolution{W: 1600, H: 1200}, com7ResUXGA}} {
		b := init(sensor.Mode{Resolution: c.res, Format: sensor.RGB565, TestPattern: true}, Options{})
		if sen(b, com7) != c.com7|com7ColorBar {
			t.Errorf("%v colour bars: COM7 %#x, want %#x", c.res, sen(b, com7), c.com7|com7ColorBar)
		}
	}
	if b := init(sensor.Mode{Resolution: sensor.QVGA, Format: sensor.RGB565}, Options{JPEGSensorHREF: true}); dsp(b, imageMode) != imageModeRGB565 {
		t.Errorf("RGB565 with JPEGSensorHREF: IMAGE_MODE %#x, want RGB565", dsp(b, imageMode))
	}
}
