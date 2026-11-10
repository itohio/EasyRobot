// Package psram drives the 8 MB octal PSRAM inside the ESP32-S3R8 (APMemory
// or UniIC, 3.3 V, on MSPI chip-select CS1) under TinyGo, without ESP-IDF.
//
// Init (esp32s3 only) configures the PSRAM's mode registers over SPI1,
// programs the SPI0 cache-access phases for octal DTR, and maps the memory
// into the data bus through the MMU. The memory is a fixed region outside the
// Go heap: the GC never scans or moves it. Allocate from it with an Arena;
// keep DMA buffers cache-line aligned and call WritebackInvalidate /
// Invalidate around GDMA transfers: the GDMA writes PSRAM behind the D-cache
// (shown on the board, ER-ESP32CAM-E7 evidence, outside the repo).
//
// The init sequence follows ESP-IDF v5.3.1 esp_psram_impl_octal.c
// (Apache-2.0) at a conservative 40 MHz PSRAM clock (no timing tuning needed).
// This file holds the hardware-independent parts, tested on the host.
package psram

import "unsafe"

// Error is a package error without allocation.
type Error uint8

const (
	ErrNotFound    Error = iota + 1 // vendor id is neither APMemory (0x0D) nor UniIC (0x1A)
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
	// the flash image (code and rodata) through the low entries from 0; PSRAM
	// takes the top entries and at most the upper half, well clear of them.
	mmuTable       = 0x600C5000
	dbusBase       = 0x3C000000
	dbusFirstEntry = 256 // lowest entry PSRAM may use (upper half of the table)
	mmuEntries     = 512
	mmuAccessPSRAM = 1 << 15 // SOC_MMU_ACCESS_SPIRAM
	mmuInvalid     = 1 << 14

	vendorAPMemory = 0x0D
	vendorUniIC    = 0x1A
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

// IO MUX: one register per GPIO from 0x60009004 (IO_MUX_GPIO0_REG).
const (
	ioMuxBase = 0x60009000
	cs1GPIO   = 26 // SPICS1: PSRAM chip select (IO MUX function 0)
)

// mspiDataPins are the MSPI pads besides CLK/CS: SPIHD, SPIWP, SPIQ, SPID,
// SPIIO4–7 and SPIDQS. Octal PSRAM uses all of them; ESP-IDF's
// esp_mspi_pin_init routes them with the ROM's esp_rom_opiflash_pin_config
// and raises their drive to 3 (spi_timing_set_pin_drive_strength).
var mspiDataPins = [...]int{27, 28, 31, 32, 33, 34, 35, 36, 37}

// ioMuxAddr is the IO MUX register of gpio.
func ioMuxAddr(gpio int) uintptr { return ioMuxBase + 4 + uintptr(gpio)*4 }

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

// knownVendor reports a PSRAM vendor this driver accepts: APMemory and UniIC,
// the two ESP-IDF 5.5.5 accepts for octal PSRAM (4.4.7: APMemory only).
func knownVendor(v uint8) bool { return v == vendorAPMemory || v == vendorUniIC }

// Region is the mapped PSRAM. base is an unsafe.Pointer (not a uintptr) so
// that slices are made by pointer arithmetic; the memory is outside the Go
// heap and the GC never follows it.
type Region struct {
	base unsafe.Pointer
	size uint32
}

func (r Region) Base() uintptr { return uintptr(r.base) }

// Bytes returns the whole region as a byte slice. It is not GC-managed
// memory: the GC neither scans nor frees it, so store no Go pointers in it.
func (r Region) Bytes() []byte {
	if r.size == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(r.base), r.size)
}

// syncRange returns the address and size the cache operations act on for b:
// len(b) rounded up to whole cache lines, never the rest of cap(b) (a slice
// of Bytes() keeps the region's capacity). b must start on a cache line, and
// its capacity must own the rounded-up last line, or another value sharing
// that line could lose its cached writes (ErrAlign). An empty b is a no-op.
func (r Region) syncRange(b []byte) (uint32, uint32, error) {
	if r.size == 0 {
		return 0, 0, ErrNoInit
	}
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	n := (len(b) + CacheLine - 1) &^ (CacheLine - 1)
	if n == 0 {
		return uint32(p), 0, nil
	}
	if !r.contains(p, n) {
		return 0, 0, ErrBounds
	}
	if p%CacheLine != 0 || n > cap(b) {
		return 0, 0, ErrAlign
	}
	return uint32(p), uint32(n), nil
}
func (r Region) Size() uint32 { return r.size }

// contains reports whether [addr, addr+n) lies inside the region.
func (r Region) contains(addr uintptr, n int) bool {
	base := uintptr(r.base)
	return n >= 0 && addr >= base && uint64(addr-base)+uint64(n) <= uint64(r.size)
}

// Arena hands out cache-line aligned, cache-line sized pieces of a Region.
// There is no free: pieces live as long as the program (frame buffers).
type Arena struct {
	r    Region
	next uint32 // offset of the next free byte
}

func NewArena(r Region) Arena { return Arena{r: r} }

// reserve returns the region offset of n bytes rounded up to cache lines.
func (a *Arena) reserve(n int) (uint32, int, error) {
	if a.r.size == 0 {
		return 0, 0, ErrNoInit
	}
	rounded := (n + CacheLine - 1) &^ (CacheLine - 1)
	if n <= 0 || uint64(a.next)+uint64(rounded) > uint64(a.r.size) {
		return 0, 0, ErrFull
	}
	off := a.next
	a.next += uint32(rounded)
	return off, rounded, nil
}

// Alloc returns n bytes from the arena: cache-line aligned start, length n,
// capacity rounded up to whole cache lines, so the cache operations on it
// never touch another allocation's lines.
func (a *Arena) Alloc(n int) ([]byte, error) {
	off, rounded, err := a.reserve(n)
	if err != nil {
		return nil, err
	}
	return unsafe.Slice((*byte)(unsafe.Add(a.r.base, off)), rounded)[:n:rounded], nil
}

// Used returns the bytes handed out so far.
func (a *Arena) Used() uint32 { return a.next }

// Reset makes the whole region available again (callers must drop old slices).
func (a *Arena) Reset() { a.next = 0 }
