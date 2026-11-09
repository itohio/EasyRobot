---
task: ER-ESP32CAM
milestone: E0 Identify (read-only)
owner: esp32cam
date: 2026-11-09
branch: esp32cam (worktree ~/src/worktrees/EasyRobot-esp32cam, from db9eb8f)
---

# ER-ESP32CAM E0 — board, sensor, TinyGo support, backend layout

**Result:** the board is a **Seeed XIAO ESP32-S3 Sense** (ESP32-S3R8, 8 MB in-package PSRAM, 8 MB GigaDevice quad flash). TinyGo 0.43 builds and runs on it (E1). It has the USB Serial/JTAG console, I2C (enough for SCCB), register structs for LCD_CAM, and a working GDMA descriptor-ring user: the I2S driver (correction after verify, see §3). It has **no camera driver and no PSRAM support**. The sensor type (OV2640, OV3660 or OV5640) is not readable without firmware: E2 probes it over SCCB.

## 1. Observed on `/dev/cu.usbmodem101` (all read-only)

| Item | Value | Source |
|---|---|---|
| Chip | ESP32-S3 (QFN56) rev v0.2, dual core 240 MHz, Wi-Fi + BT5 LE | `esptool chip-id` |
| PSRAM | Embedded 8 MB (AP_3v3) → ESP32-S3**R8** die | `esptool chip-id` "Features" |
| Crystal | 40 MHz | `esptool chip-id` |
| MAC | 1c:db:d4:74:55:80 (matches the peer file's USB serial) | `esptool chip-id` |
| USB | VID 0x303A, PID 0x1001, bcdDevice 0x0101, "Espressif / USB JTAG/serial debug unit", serial `1C:DB:D4:74:55:80` — the chip's built-in USB Serial/JTAG, no USB-UART bridge | `ioreg -p IOUSB -l` |
| Flash | JEDEC c8/4017 = GigaDevice GD25Q64-class, 8 MB, quad, 3.3 V (eFuse) | `esptool flash-id` |
| Security | Secure boot off, flash encryption off, key blocks empty | `esptool get-security-info` |
| Factory boot log | ROM `esp32s3-20210327`, `SPI_FAST_FLASH_BOOT`, then `Hello from Seeed Studio XIAO ESP32-S3 Sense` and a row of dots | `tmp/esp32cam/mon.py 8 --reset` → `tmp/esp32cam/factory-boot.log` |
| Partition table (0x8000) | nvs 0x9000/0x5000, otadata 0xe000/0x2000, app0 0x10000/0x330000, app1 0x340000/0x330000, spiffs 0x670000/0x180000, coredump 0x7f0000/0x10000 (Arduino default 8 MB layout) | parsed from the flash backup (E1) |

Board identification: the boot banner names it, and the chip/flash match Seeed's XIAO ESP32-S3 (S3R8 + 8 MB flash). The "Sense" part is the camera/microSD/mic daughterboard on the B2B connector.

## 2. Camera hardware (reported, not yet measured)

Sources (read-only web, disclosed): Seeed wiki source `Seeed-Studio/OSHW-XIAO-Series/document/SeeedStudio_XIAO_ESP32S3_Sense/XIAO_ESP32S3_Sense_camera.md` (vendor, primary for pins); `developing-today/hardware-doc` `xiao-esp32s3-sense/features/camera.md` (third party, schematic-based, for sensor variants); ESP-IDF v5.3 `soc/esp32s3/include/soc/gpio_sig_map.h` (primary, signal indices).

| Signal | GPIO | GPIO-matrix signal (S3) |
|---|---|---|
| XCLK (XMCLK) | 10 | `CAM_CLK_IDX` 149 (out) — LCD_CAM can generate XCLK itself; LEDC is the alternative |
| PCLK | 13 | `CAM_PCLK_IDX` 149 (in) |
| VSYNC | 38 | `CAM_V_SYNC_IDX` 152 |
| HREF | 47 | `CAM_H_ENABLE_IDX` 150 |
| D0..D7 (Y2..Y9) | 15, 17, 18, 16, 14, 12, 11, 48 | `CAM_DATA_IN0..7_IDX` 133..140 |
| SCCB SCL / SDA | 39 / 40 | I2C0 89 / 90 |
| PWDN, RESET | not connected (−1 in Seeed's pin map) | — |

Sensor: Seeed has shipped **OV2640** (early units, SCCB 0x30, 8-bit registers, PID 0x26), **OV3660** (current default, SCCB 0x3C, 16-bit registers, PID 0x3660) and **OV5640** (paid upgrade, 0x3C, 16-bit, PID 0x5640) on this board (reported). All three output JPEG, YUV422, RGB565 over 8-bit DVP. E2 reads the ID; the owner can also read the module label.

## 3. TinyGo 0.43 support for the ESP32-S3 (read from the installed tree `/opt/homebrew/Cellar/tinygo/0.43.0`)

| Need | Status | Evidence |
|---|---|---|
| Target | `xiao-esp32s3` (inherits `esp32s3`, tag `xiao_esp32s3`), `esp32s3-generic` | `targets/xiao-esp32s3.json`, `targets/esp32s3.json` |
| Console | `"serial": "usb"` → `machine.USBCDC` on the USB Serial/JTAG controller (hardware CDC-ACM) | `src/machine/machine_esp32xx_usb.go`; observed working in E1 |
| Board pins | D0–D10, LED GPIO21, UART 43/44, I2C 5/6, SPI 7/8/9; no camera pins | `src/machine/board_xiao-esp32s3.go` |
| I2C (for SCCB) | `machine.I2C0/I2C1` via the GPIO matrix, any pins | `machine_esp32s3_i2c.go`, `machine_esp32xx_i2c.go` |
| PWM (XCLK fallback) | LEDC driver | `machine_esp32s3_pwm.go` |
| LCD_CAM camera | **register structs only** (`esp.LCD_CAM` 0x60041000, `CAM_CTRL/CTRL1/RGB_YUV`, `IRQ_LCD_CAM` 24); no machine driver | `src/device/esp/esp32s3.go:97,571,31007` |
| GDMA | register structs (`esp.DMA` 0x6003F000, `IN_LINK_CHn`, `IN_PERI_SEL_CHn`), **and the I2S driver already runs GDMA descriptor rings on channel 0**: `src/machine/machine_esp32xx_i2s.go` (`i2sDescriptor`, ring of 3, `nextTx`/`nextRx`), `machine_esp32s3_i2s.go` (channel-0 register map, DMA clock/reset, `MISC_CONF.CLK_EN`). No generic DMA API. *Corrected 2026-11-09: the first version said "no driver", which was wrong (verify, via Leader).* This is the in-tree reference for the camera DMA. | `src/device/esp/esp32s3.go:529,3725–3743`; files named |
| PSRAM | **not supported**: the linker script maps only internal SRAM1 (DRAM 416 KB at 0x3FC88000; heap ends at 0x3FCEB710); the runtime does no PSRAM/MMU init | `targets/esp32s3.ld:16–23,160–172`, `src/runtime/runtime_esp32s3.go` |
| Wi-Fi | espradio; the esp32s3 target sets build tag `espradio` by default. Owner: no Wi-Fi at all. E3 §7 shows no radio code lands in these images | `targets/esp32s3.json` |
| Flash image | `binary-format: esp32s3`: one ROM-loaded image written at offset 0x0 (header `e9`, chip id 9); `flash-method: esp32jtag` | `targets/esp32s3.json`; `hello.bin` header in E1 |

**Feasibility verdict:** a pure-TinyGo camera backend is register work like the K210's DVP, plus a GDMA descriptor chain: GPIO-matrix routing, LCD_CAM in camera mode (VSYNC/HREF framing, XCLK from `CAM_CLK`), a GDMA RX channel with `IN_PERI_SEL = LCD_CAM`, and DMA-capable buffers in internal SRAM. Without PSRAM the frame must fit in the internal heap, **measured in E1 at 385,353 B free** (heapSys 393,609 B): QVGA grayscale/Y 76.8 KB, QVGA YUV422/RGB565 153.6 KB, or a sensor-encoded JPEG (VGA/SVGA ≈ 20–60 KB, `[INFERENCE]` scene-dependent) all fit. QVGA YUV422 (Y plane → PNG on the host) or JPEG is therefore the first target for E3; PSRAM can wait. No gap found yet that forces non-TinyGo code; the risk is GDMA/LCD_CAM bring-up without a Go reference driver (ESP-IDF's `esp32-camera` `ll_cam_esp32s3.c` is the register reference, Apache-2.0, `#verify` licence when read).

## 4. Proposed `drivers/camera` backend layout (to align with k210's ER-K210 M0)

`drivers/camera` does not exist yet on `db9eb8f`; k210 owns the shared layer and interface. Proposal for the split (k210 decides the shared names; I adapt by rebasing):

```
drivers/camera/                      package camera
  camera.go         [k210]  Config{Size, Format}, Frame metadata, Backend interface:
                            Configure(Config) error; Capture(buf []byte) (n int, err error)
  sccb.go           [k210]  SCCB over x/devices.I2C; must support 8-bit (OV2640/OV7740)
                            and 16-bit register addresses (OV3660/OV5640 use 16-bit → Tx, not ReadRegister)
  ov2640.go ov7740.go ...   [k210]  shared sensor tables; OV3660/OV5640 tables proposed by esp32cam via Leader
  dvp_k210.go       [k210]  //go:build k210
  lcdcam_esp32s3.go [esp32cam] //go:build esp32s3  LCD_CAM camera mode + GPIO-matrix routing + CAM_CLK XCLK
  gdma_esp32s3.go   [esp32cam] //go:build esp32s3  GDMA RX channel, descriptor chain, EOF handling
  board_xiao_esp32s3_sense.go [esp32cam] //go:build xiao_esp32s3  pin map (§2)
```

The firmware entry point for E3 follows the repo's node convention (`drivers/lidar/main_rp2040.go`: a `main` package per node, target by build tag). The frame link uses the binary framing with CRC that k210 also needs (one shared host tool that writes PNG). Where the node `main` and the host tool live, and who owns them, is a decision for Leader (below).

## 5. Decisions for the owner / Leader

1. **Host fetch tool and frame framing:** one shared host tool (e.g. `cmd/camgrab`) for both cameras, owned by k210 with esp32cam contributing, or one per peer. Recommendation: one shared tool and one framing spec, owned by k210 (it owns the shared layer); esp32cam reuses it.
2. **ESP32 node `main` location:** `drivers/camera/cmd/esp32s3/` (inside my write scope if Leader extends it) or the throwaway probe in `tmp/esp32cam/` until k210's interface lands. Recommendation: probes in `_scratch` for E2, the node under `drivers/camera/` once k210 commits the interface.

## 6. Unverified / assumptions

- Sensor model: unknown until E2 (`#verify`).
- Camera pin map and PWDN/RESET unconnected: vendor docs only until E2/E3 exercise them.
- Heap for frame buffers: 385,353 B free under TinyGo 0.43 (E1, `runtime.MemStats`); DMA-capability of every heap address is assumed (all of SRAM1 is DMA-reachable per the S3 TRM, `#verify`).

## 7. Network contact (disclosed)

Read-only: one web search (XIAO ESP32-S3 Sense camera pins); fetched `raw.githubusercontent.com` Seeed-Studio/OSHW-XIAO-Series camera doc, developing-today/hardware-doc camera.md, espressif/esp-idf v5.3 `gpio_sig_map.h` (curl, filtered to CAM/I2C lines). No downloads kept.
