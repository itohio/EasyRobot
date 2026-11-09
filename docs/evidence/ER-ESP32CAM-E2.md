---
task: ER-ESP32CAM
milestone: E2 Sensor ID
owner: esp32cam
date: 2026-11-09
board: Seeed XIAO ESP32-S3 Sense, /dev/cu.usbmodem101, MAC 1c:db:d4:74:55:80
---

# ER-ESP32CAM E2 — sensor ID over SCCB (TinyGo)

**Result:** the sensor is an **OmniVision OV3660**: the only device on the camera SCCB bus answers at 7-bit address **0x3C**, and its 16-bit registers 0x300A/0x300B read **0x36 0x60** (CHIP_ID 0x3660), four times in a row. Nothing answers at 0x30 (OV2640).

## Probe

`~/src/tmp/esp32cam/camid/main.go` (TinyGo 0.43, `-target=xiao-esp32s3`, 18,484 B flash):

1. XCLK: LEDC `machine.PWM0`, `Period: 50` ns (20 MHz; LEDC picks 2-bit resolution, `Top()` = 3), output on GPIO10.
2. SCCB: `machine.I2C0`, SCL GPIO39, SDA GPIO40, 100 kHz.
3. Scan 0x08–0x77 with a 1-byte read; then OV2640 path (0x30: write `FF=01`, read 0x0A/0x0B/0x1C/0x1D) and OV3660/OV5640 path (0x3C: write the 16-bit register address, stop, read one byte). SCCB has no repeated start, so each read is two `Tx` calls.

```
GOTOOLCHAIN=local GOPROXY=off tinygo build -target=xiao-esp32s3 -size short -o camid.bin ./camid
   code    data     bss |   flash     ram
  18276     208    8664 |   18484    8872
esptool --port /dev/cu.usbmodem101 write-flash 0x0 camid.bin
Wrote 18576 bytes (10647 compressed) at 0x00000000 … Hash of data verified.
```

Console (`mon.py 7`, log `tmp/esp32cam/camid-run.log`):

```
entry 0x4037947c
camid run 0 xclk top 3
scan: 0x3c ( 1 devices )
0x30: no ack: i2c: error: expected ACK not NACK
0x3C 16-bit regs: CHIP_ID 0x3660 (0x3660 = OV3660, 0x5640 = OV5640)
… (runs 1–3 identical)
```

## Consequences for E3 and the shared layer

- **OV3660 uses 16-bit register addresses.** k210's proposed shared `sensor.Bus` (`ReadReg(reg uint8)`, `WriteReg(reg, val uint8)`, ER-K210-M0 §1) cannot address it. Proposal for k210 via Leader: `ReadReg(reg uint16) (uint8, error)` / `WriteReg(reg uint16, val uint8) error` with the bus told its address width at construction (8 for OV7740/OV2640, 16 for OV3660/OV5640), or a second `Bus16` interface. The frame format v1 already lists `sensor` 3 = OV3660.
- Sensor tables for E3 come from `espressif/esp32-camera` v2.1.9 (`sensors/private_include/ov3660_settings.h`, Apache-2.0), cloned read-only to `~/src/docs/research/esp32-camera` with a provenance row.
- XCLK duty is 25 % (`Set(ch, Top()/2)` with `Top()` = 3); SCCB answered anyway. E3 uses 50 % (`(Top()+1)/2`).

## State

At the time of E2 the board ran `camid` (TinyGo image at 0x0). This section first promised a factory restore after E3 and said the backups were kept. Both were dropped by the owner on 2026-11-09 ("i don't care about original firmware on the boards. no wifi required at all."):
- Leader deleted the backup images because they held Wi-Fi credentials. Only `backup/SHA256SUMS` (`ea18e0ff…37691`) remains, as the record.
- From now on there are no factory restores; test firmware stays on the board between sessions.

The one restore that did happen (end of E3, before the decision reached me) is described in E3 §5.

## Network contact (disclosed)

Read-only: shallow sparse `git clone` of https://github.com/espressif/esp32-camera (v2.1.9 `b3f7fe8`) into `docs/research/esp32-camera` for E3 reference (provenance row added to `docs/research/README.md`, per the owner's research-download rule).
