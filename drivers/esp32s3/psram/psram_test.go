package psram

import "testing"

func TestModeRegsDecode(t *testing.T) {
	// MR0 0x28 = fixed latency, read latency code 2, drive 0; MR1 vendor 0x0D;
	// MR2 0x83 = good die, density 3 (8 MB); MR3 0x40 = 3 V.
	m := ModeRegs{MR0: 0x28, MR1: 0x0D, MR2: 0x83, MR3: 0x40}
	if m.VendorID() != vendorAPMemory || !m.GoodDie() || !m.FixedLatency() || m.ReadLatency() != 2 ||
		m.DriveStrength() != 0 || !m.VCC3V() || m.Size() != 8<<20 {
		t.Fatalf("decode %+v: vendor %x good %v fixed %v rl %d ds %d 3v %v size %d", m, m.VendorID(), m.GoodDie(),
			m.FixedLatency(), m.ReadLatency(), m.DriveStrength(), m.VCC3V(), m.Size())
	}
	for code, want := range map[uint8]uint32{1: 4 << 20, 3: 8 << 20, 5: 16 << 20, 7: 32 << 20, 6: 64 << 20, 0: 0, 2: 0} {
		if got := (ModeRegs{MR2: code}).Size(); got != want {
			t.Errorf("density %d: size %d, want %d", code, got, want)
		}
	}
}

func TestMR0ValueKeepsReserved(t *testing.T) {
	for _, read := range []uint8{0x00, 0xFF, 0xC3, 0x40} {
		v := mr0Value(read)
		if v&0xC0 != read&0xC0 {
			t.Errorf("read %02x: reserved bits changed → %02x", read, v)
		}
		m := ModeRegs{MR0: v}
		if !m.FixedLatency() || m.ReadLatency() != 2 || m.DriveStrength() != 0 {
			t.Errorf("read %02x: mr0 %02x decodes wrong", read, v)
		}
	}
}

func TestLayout8MB(t *testing.T) {
	m, ok := layout(8 << 20)
	// The MMU indexes entries by linear address, (vaddr & 0x1FFFFFF) >> 16
	// (ESP-IDF mmu_ll_get_entry_id): entries 384..511 serve 0x3D800000.
	if !ok || m.firstEntry != 384 || m.entries != 128 || m.vaddr != 0x3D800000 {
		t.Fatalf("layout 8 MB = %+v %v; want entries 384..511 at 0x3D800000", m, ok)
	}
	if got := int(m.vaddr&0x1FFFFFF) / PageSize; got != m.firstEntry {
		t.Errorf("vaddr 0x%x is served by entry %d, not %d", m.vaddr, got, m.firstEntry)
	}
	// The mapped range ends exactly at the end of the DRAM0 cache window.
	if end := m.vaddr + uintptr(m.entries)*PageSize; end != 0x3E000000 {
		t.Errorf("end 0x%x", end)
	}
	if mmuAddr(m.firstEntry) != 0x600C5600 || mmuAddr(511) != 0x600C57FC {
		t.Errorf("entry addresses 0x%x 0x%x", mmuAddr(m.firstEntry), mmuAddr(511))
	}
	if mmuValue(0) != 0x8000 || mmuValue(127) != 0x807F || mmuValue(1)&mmuInvalid != 0 {
		t.Errorf("mmu values")
	}
}

func TestLayoutLimits(t *testing.T) {
	if _, ok := layout(0); ok {
		t.Error("0 bytes mapped")
	}
	if m, ok := layout(16 << 20); !ok || m.firstEntry != 256 || m.vaddr != 0x3D000000 {
		t.Errorf("16 MB: %+v %v", m, ok)
	}
	if _, ok := layout(16<<20 + 1); ok {
		t.Error("more than the data-bus half mapped")
	}
	if m, _ := layout(PageSize + 1); m.entries != 2 {
		t.Errorf("partial page rounds up: %d entries", m.entries)
	}
}

func TestSclk(t *testing.T) {
	cases := []struct{ div, want uint32 }{
		{1, 1 << 31},
		{2, 1<<16 | 0<<8 | 1}, // 80 MHz / 2 = 40 MHz
		{4, 3<<16 | 1<<8 | 3},
	}
	for _, c := range cases {
		if got := sclk(c.div); got != c.want {
			t.Errorf("sclk(%d) = %08x, want %08x", c.div, got, c.want)
		}
	}
}

func TestPhaseConstants(t *testing.T) {
	if rdDummyBits != 18 || wrDummyBits != 8 || cmdBits != 16 || addrBits != 32 {
		t.Fatal("phase lengths changed")
	}
}

func TestArena(t *testing.T) {
	r := Region{base: 0x3C800000, size: 8 << 20}
	a := NewArena(r)
	p, n, err := a.reserve(100)
	if err != nil || p != 0x3C800000 || n != 128 {
		t.Fatalf("first: %x %d %v", p, n, err)
	}
	p2, _, _ := a.reserve(1)
	if p2 != 0x3C800080 || !alignedRange(p2, 32) {
		t.Fatalf("second not aligned: %x", p2)
	}
	if _, _, err := a.reserve(8 << 20); err != ErrFull {
		t.Errorf("overflow: %v", err)
	}
	if _, _, err := a.reserve(0); err != ErrFull {
		t.Errorf("zero size: %v", err)
	}
	if a.Used() != 160 {
		t.Errorf("used %d", a.Used())
	}
	a.Reset()
	if p3, _, _ := a.reserve(32); p3 != r.base {
		t.Errorf("reset: %x", p3)
	}
	if a0 := NewArena(Region{}); func() error { _, _, e := a0.reserve(32); return e }() != ErrNoInit {
		t.Errorf("uninitialised region: %v", err)
	}
}

func TestRegionContains(t *testing.T) {
	r := Region{base: 0x1000, size: 0x100}
	for _, c := range []struct {
		addr uintptr
		n    int
		want bool
	}{{0x1000, 0x100, true}, {0x10FF, 1, true}, {0x10FF, 2, false}, {0xFFF, 1, false}, {0x1000, -1, false}} {
		if got := r.contains(c.addr, c.n); got != c.want {
			t.Errorf("contains(%x, %d) = %v", c.addr, c.n, got)
		}
	}
}

func TestErrorText(t *testing.T) {
	if ErrFull.Error() != "psram: arena full" || ErrNotFound.Error() == "" {
		t.Error("error text")
	}
}

// MR8 (ESP-IDF opi_psram_mode_reg_t): bl = bits 1:0 (0 16 B, 1 32 B, 2 64 B,
// 3 1 KB), bt = bit 2 (hybrid wrap).
func TestMR8Decode(t *testing.T) {
	for _, tt := range []struct {
		mr8    uint8
		burst  int
		hybrid bool
	}{{0x00, 16, false}, {0x01, 32, false}, {0x02, 64, false}, {0x03, 1024, false}, {0x05, 32, true}, {0xFB, 1024, false}} {
		m := ModeRegs{MR8: tt.mr8}
		if m.BurstLength() != tt.burst || m.HybridWrap() != tt.hybrid {
			t.Errorf("MR8 %02x: burst %d hybrid %v, want %d %v", tt.mr8, m.BurstLength(), m.HybridWrap(), tt.burst, tt.hybrid)
		}
	}
}

// ESP-IDF's spi_timing_set_pin_drive_strength (octal MSPI, disassembled from
// the local Arduino-ESP32 SDK dio_opi libspi_flash.a) raises the drive of
// these nine IO MUX registers: SPIHD, SPIWP, SPIQ, SPID, SPIIO4–7, SPIDQS.
func TestMSPIPinRegisters(t *testing.T) {
	want := [...]uintptr{0x60009070, 0x60009074, 0x60009080, 0x60009084, 0x60009088, 0x6000908C, 0x60009090, 0x60009094, 0x60009098}
	if len(mspiDataPins) != len(want) {
		t.Fatalf("%d pins, want %d", len(mspiDataPins), len(want))
	}
	for i, p := range mspiDataPins {
		if a := ioMuxAddr(p); a != want[i] {
			t.Errorf("GPIO%d: IO MUX register 0x%x, want 0x%x", p, a, want[i])
		}
	}
	if ioMuxAddr(cs1GPIO) != 0x6000906C {
		t.Errorf("CS1 (GPIO26) IO MUX register 0x%x, want 0x6000906C", ioMuxAddr(cs1GPIO))
	}
}
