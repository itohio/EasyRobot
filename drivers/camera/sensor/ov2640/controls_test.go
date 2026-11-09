package ov2640

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// regBus is a banked register file that answers reads with what was written.
type regBus struct {
	bank uint8
	regs map[[2]uint8]uint8
}

func (b *regBus) WriteReg8(dev, reg, val uint8) error {
	if reg == bankSel {
		b.bank = val
		return nil
	}
	if b.regs == nil {
		b.regs = map[[2]uint8]uint8{}
	}
	b.regs[[2]uint8{b.bank, reg}] = val
	return nil
}
func (b *regBus) ReadReg8(dev, reg uint8) (uint8, error)            { return b.regs[[2]uint8{b.bank, reg}], nil }
func (b *regBus) ReadReg16(dev uint8, reg uint16) (uint8, error)    { return 0, nil }
func (b *regBus) WriteReg16(dev uint8, reg uint16, val uint8) error { return nil }

func TestGainEncoding(t *testing.T) {
	for _, tt := range []struct {
		g   float32
		reg uint8
	}{
		{1, 0x00}, {1.5, 0x08}, {2, 0x10}, {3, 0x18}, {4, 0x30}, {8, 0x70}, {16, 0xF0}, {31, 0xFF}, {64, 0xFF}, {0.5, 0x00},
		{3.97, 0x30}, {1.98, 0x10}, {1.47, 0x08}, // round to nearest, carrying into the next ×2 stage
	} {
		if got := encodeGain(tt.g); got != tt.reg {
			t.Errorf("encodeGain(%v) = %#x, want %#x", tt.g, got, tt.reg)
		}
	}
	if g := decodeGain(0xFF); g != 31 {
		t.Errorf("decodeGain(0xFF) = %v, want 31", g)
	}
}

func TestWhiteBalanceGainClamps(t *testing.T) {
	nan := float32(0)
	nan = nan / nan
	for _, tt := range []struct {
		g   float32
		reg uint8
	}{{1, 0x40}, {2, 0x80}, {0, 0}, {-1, 0}, {nan, 0}, {3.99, 0xFF}, {100, 0xFF}} {
		if got := wbReg(tt.g); got != tt.reg {
			t.Errorf("wbReg(%v) = %#x, want %#x", tt.g, got, tt.reg)
		}
	}
}

func TestSetControlsMixedAutoAndLongExposure(t *testing.T) {
	bus := &regBus{}
	s := &Sensor{Bus: bus, Opts: Options{LineNS: 50_000}}
	// Manual exposure above 1024 lines (needs REG45), auto gain, manual WB.
	set := sensor.Controls{ExposureUS: 100_000, AutoGain: true, WB: [3]float32{1, 1.25, 0.5}}
	if err := s.SetControls(set); err != nil {
		t.Fatal(err)
	}
	got, _ := s.ReadControls()
	if got.AutoExposure || !got.AutoGain || got.AutoWB || got.ExposureUS != 100_000 || got.WB != [3]float32{1, 1.25, 0.5} {
		t.Fatalf("read back %+v", got)
	}
	if r45 := bus.regs[[2]uint8{bankSensor, reg45}]; r45 != 2000>>10 {
		t.Fatalf("REG45 = %#x, want the top bits of 2000 lines", r45)
	}
	if _, ok := bus.regs[[2]uint8{bankSensor, gain}]; ok {
		t.Fatal("auto gain wrote the GAIN register")
	}
}

func TestSetControlsDoesNotAllocate(t *testing.T) {
	s := &Sensor{Bus: &regBus{regs: map[[2]uint8]uint8{}}, Opts: Options{LineNS: 50_000}}
	set := sensor.Controls{ExposureUS: 10_000, Gain: 2, WB: [3]float32{1, 1, 1}}
	if n := testing.AllocsPerRun(20, func() { s.SetControls(set) }); n != 0 {
		t.Fatalf("SetControls allocates %v times per call", n)
	}
}

func TestControlsRoundTrip(t *testing.T) {
	bus := &regBus{}
	// Bits the controls must not disturb.
	bus.regs = map[[2]uint8]uint8{
		{bankSensor, com8}: 0xE5, {bankSensor, aecLo}: 0x28, {bankSensor, reg45}: 0xC0, {bankDSP, ctrl1}: 0xFD,
	}
	s := &Sensor{Bus: bus, Opts: Options{LineNS: 50_000}} // 50 µs per line
	set := sensor.Controls{ExposureUS: 20_000, Gain: 4, WB: [3]float32{1.5, 1, 2}}
	if err := s.SetControls(set); err != nil {
		t.Fatal(err)
	}
	got, err := s.ReadControls()
	if err != nil {
		t.Fatal(err)
	}
	want := sensor.Controls{ExposureUS: 20_000, Gain: 4, WB: [3]float32{1.5, 1, 2}}
	if got != want {
		t.Fatalf("read back %+v, want %+v", got, want)
	}
	r := bus.regs
	if r[[2]uint8{bankSensor, com8}] != 0xE0 || r[[2]uint8{bankSensor, aecLo}]&^3 != 0x28 || r[[2]uint8{bankSensor, reg45}]&^0x3F != 0xC0 || r[[2]uint8{bankDSP, ctrl1}] != 0xF5 {
		t.Fatalf("unrelated bits changed: com8 %#x reg04 %#x reg45 %#x ctrl1 %#x",
			r[[2]uint8{bankSensor, com8}], r[[2]uint8{bankSensor, aecLo}], r[[2]uint8{bankSensor, reg45}], r[[2]uint8{bankDSP, ctrl1}])
	}
	// 20 ms / 50 µs = 400 lines = 0x190: AEC[1:0]=0, AEC[9:2]=0x64, AEC[15:10]=0.
	if r[[2]uint8{bankSensor, aec}] != 0x64 || r[[2]uint8{bankSensor, aecLo}]&3 != 0 {
		t.Fatalf("AEC registers %#x/%#x, want 400 lines", r[[2]uint8{bankSensor, aec}], r[[2]uint8{bankSensor, aecLo}])
	}
}

func TestControlsAutoKeepsManualRegisters(t *testing.T) {
	bus := &regBus{regs: map[[2]uint8]uint8{{bankSensor, aec}: 0x33, {bankSensor, gain}: 0x12}}
	s := &Sensor{Bus: bus} // no line period needed for auto
	if err := s.SetControls(sensor.Controls{AutoExposure: true, AutoGain: true, AutoWB: true}); err != nil {
		t.Fatal(err)
	}
	c, _ := s.ReadControls()
	if !c.AutoExposure || !c.AutoGain || !c.AutoWB {
		t.Fatalf("auto flags not set: %+v", c)
	}
	if bus.regs[[2]uint8{bankSensor, aec}] != 0x33 || bus.regs[[2]uint8{bankSensor, gain}] != 0x12 {
		t.Fatal("auto mode overwrote the sensor's exposure/gain registers")
	}
	if math.Abs(float64(c.Gain)-float64(decodeGain(0x12))) > 1e-6 {
		t.Fatalf("read-back gain %v", c.Gain)
	}
}

func TestManualExposureNeedsLineTime(t *testing.T) {
	s := &Sensor{Bus: &regBus{}}
	if err := s.SetControls(sensor.Controls{ExposureUS: 1000}); !errors.Is(err, ErrNoLineTime) {
		t.Fatalf("err = %v, want ErrNoLineTime", err)
	}
}
