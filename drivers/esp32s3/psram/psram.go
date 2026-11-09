// Package psram drives the 8 MB octal PSRAM inside the ESP32-S3R8 (APMemory,
// 3.3 V, on MSPI chip-select CS1) under TinyGo, without ESP-IDF.
//
// Init (esp32s3 only) configures the PSRAM's mode registers over SPI1,
// programs the SPI0 cache-access phases for octal DTR, and maps the memory
// into the data bus through the MMU. The memory is a fixed region outside the
// Go heap: the GC never scans or moves it. Allocate from it with an Arena;
// keep DMA buffers cache-line aligned and call WritebackInvalidate /
// Invalidate around GDMA transfers (GDMA reaches PSRAM through the D-cache).
//
// The init sequence follows ESP-IDF v5.3.1 esp_psram_impl_octal.c
// (Apache-2.0) at a conservative 40 MHz PSRAM clock (no timing tuning needed).
// This file holds the hardware-independent parts, tested on the host.
package psram

// Error is a package error without allocation.
type Error uint8

const (
	ErrNotFound    Error = iota + 1 // vendor id is not APMemory (0x0D)
	ErrUnsupported                  // unknown density code
	ErrBounds                       // offset/size outside the region
	ErrAlign                        // address or size not cache-line aligned
	ErrNoInit                       // Init has not succeeded
	ErrFull                         // arena exhausted
)

var errText = [...]string{"", "psram: chip not found (vendor id)", "psram: unsupported density",
	"psram: out of bounds", "psram: not cache-line aligned", "psram: not initialised", "psram: arena full"}

func (e Error) Error() string { return errText[e] }

const (
	CacheLine = 32      // D-cache line size configured by TinyGo's startup (esp32s3.S)
	PageSize  = 0x10000 // MMU page, 64 KB

	// MMU: 512 entries of 64 KB, indexed by linear address
	// ((vaddr & 0x1FFFFFF) >> 16, ESP-IDF mmu_ll_get_entry_id), so the DRAM0
	// cache window 0x3C000000–0x3DFFFFFF spans all 512. TinyGo's startup maps
	// flash at entries 0.. (IROM) and 256.. (DROM); PSRAM takes the top entries.
	mmuTable       = 0x600C5000
	dbusBase       = 0x3C000000
	dbusFirstEntry = 256 // lowest entry PSRAM may use (DROM starts here)
	mmuEntries     = 512
	mmuAccessPSRAM = 1 << 15 // SOC_MMU_ACCESS_SPIRAM
	mmuInvalid     = 1 << 14

	vendorAPMemory = 0x0D
)

// ModeRegs holds the PSRAM mode registers read back over SPI1.
type ModeRegs struct {
	MR0, MR1, MR2, MR3, MR4, MR8 uint8
}

func (m ModeRegs) VendorID() uint8      { return m.MR1 & 0x1F }
func (m ModeRegs) Density() uint8       { return m.MR2 & 0x07 }
func (m ModeRegs) GoodDie() bool        { return m.MR2&0x80 != 0 }
func (m ModeRegs) FixedLatency() bool   { return m.MR0&0x20 != 0 }
func (m ModeRegs) ReadLatency() uint8   { return (m.MR0 >> 2) & 0x07 }
func (m ModeRegs) DriveStrength() uint8 { return m.MR0 & 0x03 }
func (m ModeRegs) VCC3V() bool          { return m.MR3&0x40 != 0 }

// BurstLength is the MR8 wrap burst length in bytes (bl, bits 1:0).
func (m ModeRegs) BurstLength() int { return [4]int{16, 32, 64, 1024}[m.MR8&0x03] }

// HybridWrap reports MR8's hybrid wrap (bt, bit 2; ignored with a 1 KB burst).
func (m ModeRegs) HybridWrap() bool { return m.MR8&0x04 != 0 && m.MR8&0x03 != 3 }

// Size returns the device size in bytes for the density code (0 if unknown).
func (m ModeRegs) Size() uint32 {
	switch m.Density() {
	case 0x1:
		return 4 << 20
	case 0x3:
		return 8 << 20
	case 0x5:
		return 16 << 20
	case 0x7:
		return 32 << 20
	case 0x6:
		return 64 << 20
	}
	return 0
}

// mr0Value returns MR0 with fixed latency, read latency code 2 (10 cycles) and
// drive strength code 0, keeping the reserved bits of the value read.
func mr0Value(read uint8) uint8 {
	return read&^0x3F | 1<<5 | 2<<2 | 0
}

// mapping describes where n bytes of PSRAM appear on the data bus.
type mapping struct {
	firstEntry, entries int
	vaddr               uintptr
}

// layout returns the MMU entries for size bytes placed at the top of the
// table, and the data-bus address they serve.
func layout(size uint32) (mapping, bool) {
	n := int((size + PageSize - 1) / PageSize)
	if n == 0 || n > mmuEntries-dbusFirstEntry {
		return mapping{}, false
	}
	first := mmuEntries - n
	return mapping{firstEntry: first, entries: n, vaddr: dbusBase + uintptr(first)*PageSize}, true
}

// mmuValue is the table value mapping PSRAM physical page p.
func mmuValue(p int) uint32 { return uint32(p) | mmuAccessPSRAM }

// mmuAddr is the address of MMU table entry i.
func mmuAddr(i int) uintptr { return mmuTable + uintptr(i)*4 }

// sclk returns the SPI0 SRAM_CLK / SPI1 CLOCK register value dividing the
// MSPI core clock by div (div ≥ 1): EQU_SYSCLK for div 1, else
// CLKCNT_N = div−1, CLKCNT_H = div/2−1, CLKCNT_L = div−1 (fields 16, 8, 0).
func sclk(div uint32) uint32 {
	if div <= 1 {
		return 1 << 31
	}
	n := div - 1
	h := div/2 - 1
	return n<<16 | h<<8 | n
}

// Phase constants for SPI0 cache access to octal PSRAM (OPI DTR).
const (
	cmdSyncRead  = 0x0000
	cmdSyncWrite = 0x8080
	cmdRegRead   = 0x4040
	cmdRegWrite  = 0xC0C0
	cmdBits      = 16
	addrBits     = 32
	rdDummyBits  = 2 * (10 - 1) // 18
	wrDummyBits  = 2 * (5 - 1)  // 8
)

// alignedRange checks that [addr, addr+n) is cache-line aligned.
func alignedRange(addr uintptr, n int) bool {
	return addr%CacheLine == 0 && n%CacheLine == 0
}

// Region is the mapped PSRAM.
type Region struct {
	base uintptr
	size uint32
}

func (r Region) Base() uintptr { return r.base }
func (r Region) Size() uint32  { return r.size }

// contains reports whether [addr, addr+n) lies inside the region.
func (r Region) contains(addr uintptr, n int) bool {
	return n >= 0 && addr >= r.base && uint64(addr-r.base)+uint64(n) <= uint64(r.size)
}

// Arena hands out cache-line aligned, cache-line sized pieces of a Region.
// There is no free: pieces live as long as the program (frame buffers).
type Arena struct {
	r    Region
	next uintptr
}

func NewArena(r Region) Arena { return Arena{r: r, next: r.base} }

// reserve returns the address of n bytes rounded up to cache lines.
func (a *Arena) reserve(n int) (uintptr, int, error) {
	if a.r.size == 0 {
		return 0, 0, ErrNoInit
	}
	rounded := (n + CacheLine - 1) &^ (CacheLine - 1)
	if n <= 0 || !a.r.contains(a.next, rounded) {
		return 0, 0, ErrFull
	}
	p := a.next
	a.next += uintptr(rounded)
	return p, rounded, nil
}

// Used returns the bytes handed out so far.
func (a *Arena) Used() uint32 { return uint32(a.next - a.r.base) }

// Reset makes the whole region available again (callers must drop old slices).
func (a *Arena) Reset() { a.next = a.r.base }
