# psram — ESP32-S3R8 octal PSRAM for TinyGo (no ESP-IDF)

Status: build-only. The board-side test (`examples/memtest`) waits for the board.

## Use

```go
r, regs, err := psram.Init()       // once, early; maps 8 MB at 0x3D800000
a := psram.NewArena(r)             // fixed region, outside the Go heap
buf, _ := a.Alloc(300_000)         // 32-byte aligned, capacity rounded to cache lines
psram.WritebackInvalidate(buf)     // before a DMA engine writes buf
// … DMA …
psram.Invalidate(buf)              // after it, before the CPU reads buf
```

## ROM functions: build with a target that defines them

TinyGo's `esp32s3.ld` doesn't define the ROM functions this package calls. The target `xiao-esp32s3-psram.json` (in this directory) adds them with `ld --defsym`. Addresses are from ESP-IDF v5.3.1 `components/esp_rom/esp32s3/ld/esp32s3.rom.ld`:

| Symbol | Address |
|---|---|
| `esp_rom_opiflash_exec_cmd` | 0x400008b8 |
| `esp_rom_spi_set_dtr_swap_mode` | 0x4000093c |
| `rom_Cache_Suspend_DCache` | 0x400018b4 (TinyGo's `esp32s3.S` calls the same address) |
| `Cache_Resume_DCache` | 0x400018c0 (same) |
| `Cache_Invalidate_Addr` | 0x400016b0 |
| `rom_Cache_WriteBack_Addr` | 0x400016c8 |
| `Cache_Suspend_DCache_Autoload` | 0x40001734 |
| `Cache_Resume_DCache_Autoload` | 0x40001740 |

ESP-IDF wraps two of these with patches (`Cache_Suspend_DCache`, `Cache_WriteBack_Addr` in `libesp_rom.a`). Their behaviour, read from the disassembly of `esp_rom_cache.c.obj` in the local Arduino-ESP32 SDK (ESP-IDF v4.4.7, no download), is reproduced in `psram_esp32s3.go`: suspend waits until `EXTMEM CACHE_STATE.DCACHE_STATE` = 1 (idle); writeback of a cache-line aligned range runs with D-cache autoload suspended. The patch's freeze-based writeback of unaligned head/tail lines is not needed, because this package only writes back aligned ranges.

The target JSON is the `esp32s3` target with the `xiao_esp32s3` tag, without the `espradio` tag and its malloc wraps (no Wi-Fi), plus the defsyms. With the stock `-target=xiao-esp32s3` the link fails: `undefined symbol: rom_Cache_WriteBack_Addr`.

```
tinygo build -target=drivers/esp32s3/psram/xiao-esp32s3-psram.json -print-allocs=psram -o memtest.bin ./drivers/esp32s3/psram/examples/memtest   # from the repo root; the linker script path resolves inside TINYGOROOT
```

## Memory map

The S3 MMU has 512 entries of 64 KB, indexed by linear address: entry = (vaddr & 0x1FFFFFF) >> 16 (ESP-IDF `mmu_ll_get_entry_id`). The DRAM0 cache window 0x3C000000–0x3DFFFFFF therefore spans all 512 entries. TinyGo's startup maps flash at entries 0.. (IROM) and 256.. (DROM, rodata). The PSRAM takes the top 128 entries (384–511), which serve **0x3D800000–0x3DFFFFFF** (the address ESP-IDF uses too). That leaves flash rodata 8 MB, far more than any image here.

## Mode registers

`Init` writes MR0 (fixed latency, read latency 2, drive 0) and reads MR0–MR4 and MR8 back. It does not write MR8 (burst length/type): ESP-IDF v5.3.1 writes MR8 only under `CONFIG_SPIRAM_ECC_ENABLE` (`esp_psram_impl_octal.c:147-173`), and ECC is off here, as in ESP-IDF's default. `ModeRegs.BurstLength`/`HybridWrap` decode the power-on MR8; the memtest prints it, and its pattern test covers whole cache lines.

## To verify on the board

- `romModeOPIDTR = 7` (`ESP_ROM_SPIFLASH_OPI_DTR_MODE`) matches the ROM enum in the ESP32-S3 `rom/spi_flash.h` (ESP-IDF v4.4.7, checked offline by verify). A wrong value would show up as `ErrNotFound`, since the vendor id read fails.
- Init runs from flash (IROM) with the D-cache suspended only around the SPI0 phase changes and the MMU writes; no rodata is read in that window. ESP-IDF runs its init from IRAM. If init hangs, move it to IRAM.
- Register access runs at 40 MHz on SPI1. The SPI1 clock register is restored afterwards; flash access goes through SPI0 and is untouched.
