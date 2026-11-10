package ov3660

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// regBus is a register file: writes land in regs (and the write log), reads
// return regs. It starts with the OV3660 product id.
type regBus struct {
	regs   map[uint16]uint8
	writes []reg
}

func newBus() *regBus {
	return &regBus{regs: map[uint16]uint8{regPID: 0x36, regVER: 0x60}}
}

func (b *regBus) WriteReg16(dev uint8, r uint16, v uint8) error {
	if dev != Addr {
		return errors.New("wrong device")
	}
	b.regs[r] = v
	b.writes = append(b.writes, reg{r, v})
	return nil
}
func (b *regBus) ReadReg16(dev uint8, r uint16) (uint8, error) {
	if dev != Addr {
		return 0, errors.New("wrong device")
	}
	return b.regs[r], nil
}
func (b *regBus) ReadReg8(dev, r uint8) (uint8, error) { return 0, errors.New("8-bit access") }
func (b *regBus) WriteReg8(dev, r, v uint8) error      { return errors.New("8-bit access") }
func (b *regBus) get16(r uint16) uint16                { return uint16(b.regs[r])<<8 | uint16(b.regs[r+1]) }
func mode(w, h uint16, f sensor.Format) sensor.Mode {
	return sensor.Mode{Resolution: sensor.Resolution{W: w, H: h}, Format: f}
}
func crop(w, h uint16, f sensor.Format) sensor.Mode { m := mode(w, h, f); m.Crop = true; return m }

func TestInitProgramsWindowTimingFormatAndClock(t *testing.T) {
	tests := []struct {
		name           string
		m              sensor.Mode
		quality        uint8
		start, end     [2]uint16 // array window
		hts, vts       uint16
		binning, scale bool
		format00       uint8
		mult, pclkDiv  uint8
		pllCtrl3       uint8
		lineNS         uint32
	}{
		// QVGA YUV: the E3/C1 setup; line period measured 115.0 µs (C1).
		{"QVGA YUV422", mode(320, 240, sensor.YUV422), 0, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 783, true, true, 0x30, 8, 4, 0x02, 115_000},
		{"QQVGA Y8", mode(160, 120, sensor.Y8), 0, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 783, true, true, 0x10, 8, 8, 0x00, 57_500},
		{"VGA RGB565", mode(640, 480, sensor.RGB565), 0, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 783, true, true, 0x61, 4, 2, 0x02, 230_000},
		// XGA JPEG = half of 2048×1536: binned, not scaled (M3).
		{"XGA JPEG", mode(1024, 768, sensor.JPEG), 0, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 783, true, false, 0x30, 30, 10, 0x30, 46_000},
		{"UXGA JPEG q20", mode(1600, 1200, sensor.JPEG), 20, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 1564, false, true, 0x30, 30, 10, 0x30, 46_000},
		{"QXGA JPEG q40", mode(2048, 1536, sensor.JPEG), 40, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 1564, false, false, 0x30, 24, 8, 0x30, 57_500},
		{"FHD JPEG 16:9", mode(1920, 1080, sensor.JPEG), 0, [2]uint16{64, 242}, [2]uint16{2015, 1333}, 2172, 1322, false, false, 0x30, 30, 10, 0x30, 43_440},
		{"HD JPEG 16:9 binned", mode(960, 540, sensor.JPEG), 0, [2]uint16{64, 242}, [2]uint16{2015, 1333}, 2172, 662, true, false, 0x30, 30, 10, 0x30, 43_440},
		// Crop 320×240: window 352×252 centred in 2080×1548 (E3c windowed mode).
		{"crop QVGA YUV422", crop(320, 240, sensor.YUV422), 0, [2]uint16{864, 648}, [2]uint16{1215, 899}, 2300, 1564, false, false, 0x30, 8, 4, 0x02, 115_000},
		{"crop QXGA JPEG", crop(2048, 1536, sensor.JPEG), 0, [2]uint16{0, 0}, [2]uint16{2079, 1547}, 2300, 1564, false, false, 0x30, 24, 8, 0x30, 57_500},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBus()
			s := &Sensor{Bus: b, Opts: Options{Quality: tt.quality}}
			if err := s.Init(tt.m); err != nil {
				t.Fatal(err)
			}
			if b.writes[0] != (reg{regSystemCtrl0, 0x82}) {
				t.Errorf("first write %v, want soft reset", b.writes[0])
			}
			if got := [2]uint16{b.get16(regXAddrSt), b.get16(regXAddrSt + 2)}; got != tt.start {
				t.Errorf("window start %v, want %v", got, tt.start)
			}
			if got := [2]uint16{b.get16(regXAddrEnd), b.get16(regXAddrEnd + 2)}; got != tt.end {
				t.Errorf("window end %v, want %v", got, tt.end)
			}
			if w, h := b.get16(regXOutput), b.get16(regXOutput+2); w != tt.m.W || h != tt.m.H {
				t.Errorf("output %d×%d, want %d×%d", w, h, tt.m.W, tt.m.H)
			}
			if hts, vts := b.get16(regXTotal), b.get16(regXTotal+2); hts != tt.hts || vts != tt.vts {
				t.Errorf("totals %d×%d, want %d×%d", hts, vts, tt.hts, tt.vts)
			}
			binned := b.regs[regTiming20]&0x01 != 0 && b.regs[regTiming21]&0x01 != 0 && b.regs[regXInc] == 0x31 && b.regs[regYInc] == 0x31
			if binned != tt.binning {
				t.Errorf("binning %v (0x3820 %#x 0x3821 %#x inc %#x), want %v", binned, b.regs[regTiming20], b.regs[regTiming21], b.regs[regXInc], tt.binning)
			}
			wantOff := [2]uint16{16, 6}
			if tt.binning {
				wantOff = [2]uint16{8, 2}
			}
			if got := [2]uint16{b.get16(regXOffset), b.get16(regXOffset + 2)}; got != wantOff {
				t.Errorf("ISP offsets %v, want %v", got, wantOff)
			}
			if scaled := b.regs[regISPCtrl01]&isp01Scale != 0; scaled != tt.scale {
				t.Errorf("scale %v, want %v", scaled, tt.scale)
			}
			if b.regs[regFormat00] != tt.format00 {
				t.Errorf("FORMAT_CTRL00 %#x, want %#x", b.regs[regFormat00], tt.format00)
			}
			jpeg := tt.m.Format == sensor.JPEG
			if compressed := b.regs[regTiming21]&0x20 != 0; compressed != jpeg {
				t.Errorf("compression bit %v, want %v", compressed, jpeg)
			}
			if jpeg {
				q := tt.quality
				if q == 0 {
					q = 12
				}
				if b.regs[regCompress07] != q {
					t.Errorf("quality %d, want %d", b.regs[regCompress07], q)
				}
			} else if _, ok := b.regs[regCompress07]; ok {
				t.Errorf("raw format wrote COMPRESSION_CTRL07")
			}
			if b.regs[regPLLCtrl1] != tt.mult || b.regs[regPCLKRatio] != tt.pclkDiv || b.regs[regPLLCtrl3] != tt.pllCtrl3 {
				t.Errorf("PLL mult %d pclk %d ctrl3 %#x, want %d %d %#x", b.regs[regPLLCtrl1], b.regs[regPCLKRatio], b.regs[regPLLCtrl3], tt.mult, tt.pclkDiv, tt.pllCtrl3)
			}
			if s.LineNS() != tt.lineNS {
				t.Errorf("line %d ns, want %d", s.LineNS(), tt.lineNS)
			}
			if _, ok := b.regs[regTestSetting]; ok {
				t.Errorf("test pattern register written without TestPattern")
			}
		})
	}
}

func TestInitRejectsUnsupportedModes(t *testing.T) {
	for _, m := range []sensor.Mode{
		mode(0, 0, sensor.YUV422),
		mode(322, 242, sensor.YUV422),    // not multiples of 4
		mode(320, 100, sensor.YUV422),    // 16:5: no ratio entry
		mode(2560, 1920, sensor.JPEG),    // 4:3 beyond the maximum
		mode(320, 240, sensor.Format(0)), // unknown format
		mode(320, 240, sensor.Format(99)),
		crop(2052, 1536, sensor.JPEG),
		crop(320, 1540, sensor.JPEG),
	} {
		b := newBus()
		s := &Sensor{Bus: b}
		if err := s.Init(m); !errors.Is(err, ErrMode) {
			t.Errorf("%+v: err %v, want ErrMode", m, err)
		}
		if len(b.writes) != 0 {
			t.Errorf("%+v: %d writes before rejecting", m, len(b.writes))
		}
	}
}

func TestInitAcceptsAnySizeOfARatioEntry(t *testing.T) {
	for _, m := range []sensor.Mode{mode(1920, 1440, sensor.JPEG), mode(300, 200, sensor.Y8), mode(864, 1536, sensor.JPEG), mode(480, 480, sensor.YUV422)} {
		if err := (&Sensor{Bus: newBus()}).Init(m); err != nil {
			t.Errorf("%+v: %v", m, err)
		}
	}
}

func TestInitChecksProductID(t *testing.T) {
	b := newBus()
	b.regs[regPID] = 0x56 // OV5640
	b.regs[regVER] = 0x40
	if err := (&Sensor{Bus: b}).Init(mode(320, 240, sensor.YUV422)); !errors.Is(err, ErrID) {
		t.Fatalf("err %v, want ErrID", err)
	}
	if len(b.writes) != 0 {
		t.Errorf("%d writes to a non-OV3660", len(b.writes))
	}
}

func TestSupportedMatchesInit(t *testing.T) {
	for _, m := range []sensor.Mode{
		mode(320, 240, sensor.YUV422), mode(2048, 1536, sensor.JPEG), crop(320, 240, sensor.Y8),
		mode(322, 242, sensor.YUV422), mode(320, 100, sensor.JPEG), crop(2052, 1536, sensor.JPEG),
		mode(320, 240, sensor.Format(99)),
	} {
		err := (&Sensor{Bus: newBus()}).Init(m)
		if Supported(m) != (err == nil) {
			t.Errorf("%+v: Supported %v, Init %v", m, Supported(m), err)
		}
	}
}

func TestInitTestPattern(t *testing.T) {
	b := newBus()
	m := mode(320, 240, sensor.RGB565)
	m.TestPattern = true
	if err := (&Sensor{Bus: b}).Init(m); err != nil {
		t.Fatal(err)
	}
	if b.regs[regTestSetting] != testColorBar {
		t.Errorf("PRE_ISP_TEST_SETTING_1 %#x, want %#x", b.regs[regTestSetting], testColorBar)
	}
}

// esp32-camera's set_pll comments give SYSCLK for its 16 MHz tuning and the
// JPEG settings at 20 MHz.
func TestPLLSysclkMatchesEsp32Camera(t *testing.T) {
	tests := []struct {
		p    pll
		xclk uint32
		want uint32
	}{
		{pll{4, 1, 0, false, 2, 2}, 16_000_000, 8_000_000},
		{pll{8, 1, 0, false, 2, 4}, 16_000_000, 16_000_000},
		{pll{8, 1, 0, false, 0, 8}, 16_000_000, 32_000_000},
		{pll{24, 1, 3, false, 0, 8}, 20_000_000, 40_000_000},
		{pll{30, 1, 3, false, 0, 10}, 20_000_000, 50_000_000},
		{pll{8, 0, 0, true, 0, 8}, 20_000_000, 80_000_000}, // sys_div 0 counts as 1; root 2× doubles the VCO
	}
	for _, tt := range tests {
		if got := tt.p.sysclk(tt.xclk); got != tt.want {
			t.Errorf("%+v at %d Hz: SYSCLK %d, want %d", tt.p, tt.xclk, got, tt.want)
		}
	}
}

func TestLineFollowsXCLK(t *testing.T) {
	s := &Sensor{Bus: newBus(), Opts: Options{XCLKHz: 10_000_000}}
	if err := s.Init(mode(320, 240, sensor.YUV422)); err != nil {
		t.Fatal(err)
	}
	if s.LineNS() != 230_000 { // half the clock, twice the line
		t.Errorf("line %d ns at 10 MHz, want 230000", s.LineNS())
	}
}

func TestSetQuality(t *testing.T) {
	b := newBus()
	s := &Sensor{Bus: b}
	if err := s.SetQuality(40); err != nil || b.regs[regCompress07] != 40 {
		t.Errorf("SetQuality(40): err %v, register %d", err, b.regs[regCompress07])
	}
	if err := s.SetQuality(64); !errors.Is(err, ErrMode) || b.regs[regCompress07] != 40 {
		t.Errorf("SetQuality(64): err %v, register %d; want ErrMode, unchanged", err, b.regs[regCompress07])
	}
}

// One quality range for Init and SetQuality (review C5): 1 … 63, 0 = the
// default 12, above 63 refused before any register write.
func TestQualityRangeIsShared(t *testing.T) {
	b := newBus()
	s := &Sensor{Bus: b}
	if err := s.SetQuality(0); err != nil || b.regs[regCompress07] != DefaultQuality {
		t.Errorf("SetQuality(0): err %v, register %d, want the default %d", err, b.regs[regCompress07], DefaultQuality)
	}
	for _, f := range []sensor.Format{sensor.JPEG, sensor.YUV422} {
		b := newBus()
		s := &Sensor{Bus: b, Opts: Options{Quality: 64}}
		if err := s.Init(mode(320, 240, f)); !errors.Is(err, ErrMode) || len(b.writes) != 0 {
			t.Errorf("format %d, quality 64: err %v after %d writes, want ErrMode before any write", f, err, len(b.writes))
		}
	}
	b = newBus()
	s = &Sensor{Bus: b, Opts: Options{Quality: 63}}
	if err := s.Init(mode(320, 240, sensor.JPEG)); err != nil || b.regs[regCompress07] != 63 {
		t.Errorf("quality 63: err %v, register %d", err, b.regs[regCompress07])
	}
}

func near(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-3 }
