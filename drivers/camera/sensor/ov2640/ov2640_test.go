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
		{Resolution: sensor.VGA, Format: sensor.RGB565},                        // larger than CIF
		{Resolution: sensor.Resolution{W: 322, H: 240}, Format: sensor.RGB565}, // not a multiple of 4
		{Resolution: sensor.QVGA, Format: 0},
	} {
		bus := &recBus{}
		if err := (&Sensor{Bus: bus}).Init(m); !errors.Is(err, ErrMode) {
			t.Errorf("%v: err = %v, want ErrMode", m, err)
		}
	}
}
