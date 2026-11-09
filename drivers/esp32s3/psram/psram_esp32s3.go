//go:build esp32s3

package psram

import (
	"device/esp"
	"runtime/interrupt"
	"runtime/volatile"
	"unsafe"
)

// ROM functions (ESP32-S3 ROM, addresses from ESP-IDF v5.3.1
// components/esp_rom/esp32s3/ld/esp32s3.rom.ld). TinyGo's esp32s3.ld does not
// define them; the target JSON adds them with ld --defsym (see README.md).
// A body-less function with //export is an external symbol in TinyGo.

//export esp_rom_opiflash_exec_cmd
func romOpiExecCmd(spi int32, mode int32, cmd uint32, cmdBits int32, addr uint32, addrBits int32, dummyBits int32,
	mosi *byte, mosiBits int32, miso *byte, misoBits int32, csMask uint32, isWriteErase bool)

//export esp_rom_spi_set_dtr_swap_mode
func romSetDTRSwap(spi int32, wrSwap, rdSwap bool)

//export rom_Cache_Suspend_DCache
func romSuspendDCache() uint32

//export Cache_Resume_DCache
func romResumeDCache(autoload uint32)

//export Cache_Invalidate_Addr
func romInvalidate(addr, size uint32) int32

//export rom_Cache_WriteBack_Addr
func romWriteBack(addr, size uint32) int32

// romModeOPIDTR is ESP_ROM_SPIFLASH_OPI_DTR_MODE in the ROM's
// esp_rom_spiflash_read_mode_t (QIO=0 … OPI_STR=6, OPI_DTR=7). #verify on the
// board: a wrong value makes the vendor-id check below fail (ErrNotFound).
const romModeOPIDTR = 7

const (
	cs1GPIO   = 26     // SPI_CS1_GPIO_NUM on the ESP32-S3
	csMaskCS1 = 1 << 1 // BIT(1): chip-select 1
	spi1      = 1
	clkDiv    = 2 // MSPI core 80 MHz / 2 = 40 MHz PSRAM clock (no tuning needed)
)

var (
	region Region
	regs   ModeRegs
	rxBuf  [4]uint32 // ROM writes whole words
	txBuf  [4]uint32
)

func regRead(addr uint32, bits int32) uint16 {
	rxBuf[0] = 0
	romOpiExecCmd(spi1, romModeOPIDTR, cmdRegRead, cmdBits, addr, addrBits, rdDummyBits,
		nil, 0, (*byte)(unsafe.Pointer(&rxBuf[0])), bits, csMaskCS1, false)
	return uint16(rxBuf[0])
}

func regWrite(addr uint32, v uint16) {
	txBuf[0] = uint32(v)
	romOpiExecCmd(spi1, romModeOPIDTR, cmdRegWrite, cmdBits, addr, addrBits, 0,
		(*byte)(unsafe.Pointer(&txBuf[0])), 16, nil, 0, csMaskCS1, false)
}

func reg32(addr uintptr) *volatile.Register32 { return (*volatile.Register32)(unsafe.Pointer(addr)) }

// Init brings up the octal PSRAM and maps it. It runs once, early, with
// interrupts off; it must not be called while anything uses the D-cache for
// PSRAM. Returns the mapped region and the mode registers read back.
func Init() (Region, ModeRegs, error) {
	if region.size != 0 {
		return region, regs, nil
	}
	state := interrupt.Disable()
	defer interrupt.Restore(state)

	// 1. CS1 pin: IO MUX function 0 (SPICS1), drive strength 3; PSRAM clock drive 3.
	mux := (*volatile.Register32)(unsafe.Add(unsafe.Pointer(&esp.IO_MUX.GPIO0), cs1GPIO*4))
	mux.Set(mux.Get()&^(7<<12|3<<10) | 3<<10)
	esp.SPI0.SetDATE_SPI_SMEM_SPICLK_FUN_DRV(3)

	// 2. CS timing (SPI0 registers, shared with SPI1).
	esp.SPI0.SetSPI_SMEM_AC_SPI_SMEM_CS_HOLD(1)
	esp.SPI0.SetSPI_SMEM_AC_SPI_SMEM_CS_SETUP(1)
	esp.SPI0.SetSPI_SMEM_AC_SPI_SMEM_CS_HOLD_TIME(3)
	esp.SPI0.SetSPI_SMEM_AC_SPI_SMEM_CS_SETUP_TIME(3)
	esp.SPI0.SetSPI_SMEM_AC_SPI_SMEM_CS_HOLD_DELAY(2)

	// 3. ECC off (SYSCON SRAM_ACE0_ATTR bit 8).
	esp.APB_CTRL.SRAM_ACE0_ATTR.ClearBits(1 << 8)

	// 4. 40 MHz for the register phase on SPI1 and for cache access on SPI0.
	spi1Clock := esp.SPI1.CLOCK.Get()
	esp.SPI1.CLOCK.Set(sclk(clkDiv))
	esp.SPI0.SRAM_CLK.Set(sclk(clkDiv))

	// 5. SPI1 variable dummy, no DTR data swap.
	esp.SPI1.SetDDR_SPI_FMEM_VAR_DUMMY(1)
	romSetDTRSwap(spi1, false, false)

	// 6. Mode registers: MR0 = fixed latency, read latency 2, drive 0.
	mr01 := regRead(0x0, 16)
	regWrite(0x0, mr01&0xFF00|uint16(mr0Value(uint8(mr01))))
	mr01 = regRead(0x0, 16)
	mr23 := regRead(0x2, 16)
	mr4 := regRead(0x4, 8)
	mr8 := regRead(0x8, 8)
	regs = ModeRegs{MR0: uint8(mr01), MR1: uint8(mr01 >> 8), MR2: uint8(mr23), MR3: uint8(mr23 >> 8),
		MR4: uint8(mr4), MR8: uint8(mr8)}
	esp.SPI1.CLOCK.Set(spi1Clock)
	if regs.VendorID() != vendorAPMemory {
		return Region{}, regs, ErrNotFound
	}
	size := regs.Size()
	m, ok := layout(size)
	if !ok {
		return Region{}, regs, ErrUnsupported
	}

	// 7. SPI0 cache-access phases for octal DTR PSRAM. The D-cache is
	// suspended while they change; no rodata may be read in this window.
	autoload := romSuspendDCache()
	s := esp.SPI0
	s.SetCACHE_SCTRL_CACHE_SRAM_USR_WCMD(1)
	s.SetSRAM_DWR_CMD_CACHE_SRAM_USR_WR_CMD_BITLEN(cmdBits - 1)
	s.SetSRAM_DWR_CMD_CACHE_SRAM_USR_WR_CMD_VALUE(cmdSyncWrite)
	s.SetCACHE_SCTRL_CACHE_SRAM_USR_RCMD(1)
	s.SetSRAM_DRD_CMD_CACHE_SRAM_USR_RD_CMD_BITLEN(cmdBits - 1)
	s.SetSRAM_DRD_CMD_CACHE_SRAM_USR_RD_CMD_VALUE(cmdSyncRead)
	s.SetCACHE_SCTRL_SRAM_ADDR_BITLEN(addrBits - 1)
	s.SetCACHE_SCTRL_CACHE_USR_SCMD_4BYTE(1)
	s.SetCACHE_SCTRL_USR_RD_SRAM_DUMMY(1)
	s.SetCACHE_SCTRL_USR_WR_SRAM_DUMMY(1)
	s.SetCACHE_SCTRL_SRAM_RDUMMY_CYCLELEN(rdDummyBits - 1)
	s.SetSPI_SMEM_DDR_SPI_SMEM_VAR_DUMMY(1)
	s.SetCACHE_SCTRL_SRAM_WDUMMY_CYCLELEN(wrDummyBits - 1)
	s.SetSPI_SMEM_DDR_WDAT_SWP(0)
	s.SetSPI_SMEM_DDR_RDAT_SWP(0)
	s.SetSPI_SMEM_DDR_EN(1)
	s.SetSRAM_CMD_SDUMMY_OUT(1)
	s.SetSRAM_CMD_SCMD_OCT(1)
	s.SetSRAM_CMD_SADDR_OCT(1)
	s.SetSRAM_CMD_SDOUT_OCT(1)
	s.SetSRAM_CMD_SDIN_OCT(1)
	s.SetCACHE_SCTRL_SRAM_OCT(1)

	// 8. MMU: physical pages 0..n-1 in the top entries (served at m.vaddr).
	for i := range m.entries {
		reg32(mmuAddr(m.firstEntry + i)).Set(mmuValue(i))
	}
	romResumeDCache(autoload)
	romInvalidate(uint32(m.vaddr), uint32(m.entries)*PageSize)

	region = Region{base: m.vaddr, size: size}
	return region, regs, nil
}

// Bytes returns the whole region as a byte slice (not GC-managed memory).
func (r Region) Bytes() []byte {
	if r.size == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(r.base)), r.size)
}

// Alloc returns n bytes from the arena: cache-line aligned start, length n,
// capacity rounded up to whole cache lines (safe for DMA cache maintenance).
func (a *Arena) Alloc(n int) ([]byte, error) {
	p, rounded, err := a.reserve(n)
	if err != nil {
		return nil, err
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(p)), rounded)[:n:rounded], nil
}

func span(b []byte) (uintptr, int) {
	if cap(b) == 0 {
		return 0, 0
	}
	return uintptr(unsafe.Pointer(unsafe.SliceData(b))), cap(b)
}

func checked(b []byte) (uint32, uint32, error) {
	p, n := span(b)
	if region.size == 0 {
		return 0, 0, ErrNoInit
	}
	if !region.contains(p, n) {
		return 0, 0, ErrBounds
	}
	if !alignedRange(p, n) {
		return 0, 0, ErrAlign
	}
	return uint32(p), uint32(n), nil
}

// Writeback writes dirty cache lines of b (its full capacity) to PSRAM.
// Call before a DMA engine reads b.
func Writeback(b []byte) error {
	p, n, err := checked(b)
	if err == nil {
		romWriteBack(p, n)
	}
	return err
}

// Invalidate drops cached lines of b so the CPU re-reads PSRAM.
// Call after a DMA engine wrote b, before reading it.
func Invalidate(b []byte) error {
	p, n, err := checked(b)
	if err == nil {
		romInvalidate(p, n)
	}
	return err
}

// WritebackInvalidate does both; call before handing b to a DMA engine that writes it.
func WritebackInvalidate(b []byte) error {
	p, n, err := checked(b)
	if err == nil {
		romWriteBack(p, n)
		romInvalidate(p, n)
	}
	return err
}
