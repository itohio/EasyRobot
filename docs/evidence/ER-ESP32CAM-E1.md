---
task: ER-ESP32CAM
milestone: E1 Backup + hello
owner: esp32cam
date: 2026-11-09
board: Seeed XIAO ESP32-S3 Sense, /dev/cu.usbmodem101, MAC 1c:db:d4:74:55:80
---

# ER-ESP32CAM E1 — flash backup, TinyGo hello, factory restore

**Result:** full 8 MB flash backed up twice with equal SHA-256 `ea18e0ff…37691`; a TinyGo 0.43 hello ran over the USB Serial/JTAG console (heap 393,609 B, 385,353 B free); the factory image was written back, verify-flash matched the backup digest, and the factory app boots with its original banner.

Tools: esptool v5.6.0 (`/opt/homebrew/bin/esptool`), TinyGo 0.43.0 (go1.27.2, LLVM 22.1.4). Scratch dir: `~/src/tmp/esp32cam/` (`mon.py` serial capture, only ever opens `/dev/cu.usbmodem101`).

## 1. Backup

| Attempt | Method | SHA-256 | Result |
|---|---|---|---|
| a, b | two `read-flash 0 ALL`, default `--after hard-reset` | a `922e9ecc…927cd`, b `ea18e0ff…37691` | **differ**: only 0xA000–0xBFFF (NVS, 949 + 1050 bytes): the factory firmware booted between the reads and rewrote NVS |
| 1, 2 | `--after no-reset` on read 1, `--before no-reset --after no-reset` on read 2 (stub stays running, firmware never boots in between) | both `ea18e0fffcb60706adb21451ed395043aa17d492fd88aece370aae382dc37691` | **equal** (also equal to b) |

Files: `tmp/esp32cam/backup/flash-1.bin`, `flash-2.bin`, `SHA256SUMS` (a/b deleted). 8,388,608 B each, read at 921600 baud in 97 s / 79 s.

The NVS holds the factory app's Wi-Fi keys (`nvs.net80211`: `sta.ssid`, `sta.pswd`, …; values not read). **The backups may contain Wi-Fi credentials: they stay in `_scratch`, never in a repo or evidence.**

**Update 2026-11-09:** Leader deleted `flash-1.bin` and `flash-2.bin` because of those credentials, following the owner's decision ("i don't care about original firmware on the boards. no wifi required at all."). From now on there are no backups and no factory restores. `backup/SHA256SUMS` stays as the record of the equal-hash reads. A copy I recreated for the E3 restore was deleted as well (E3 §5).

## 2. TinyGo hello

`tmp/esp32cam/hello/main.go`: prints `hello esp32cam tinygo <ver> n <i> heapSys … free …` once a second on `machine.Serial` (USB CDC) and toggles LED GPIO21.

```
GOTOOLCHAIN=local GOPROXY=off tinygo build -target=xiao-esp32s3 -size short -o hello.bin ./hello
   code    data     bss |   flash     ram
   8559       8    8400 |    8567    8408
esptool --port /dev/cu.usbmodem101 write-flash --flash-mode keep --flash-freq keep --flash-size keep 0x0 hello.bin
Wrote 8672 bytes (5260 compressed) at 0x00000000 … Hash of data verified.
```

Console after reset (`mon.py 4 --reset`, log `tmp/esp32cam/hello-run2.log`):

```
rst:0x15 (USB_UART_CHIP_RESET),boot:0x2a (SPI_FAST_FLASH_BOOT)
load:0x3c000020,len:0x1b0 … load:0x42000b64,len:0x164c
SHA-256 comparison failed: … Attempting to boot anyway...
entry 0x4037947c
hello esp32cam tinygo 0.43.0 n 0 heapSys 393609 free 385353
…
hello esp32cam tinygo 0.43.0 n 4 heapSys 393609 free 385353
```

- **Heap for frame buffers: ~385 KB free** in internal SRAM (no PSRAM under TinyGo): QVGA YUV422/RGB565 (153.6 KB) fits, two QVGA buffers fit, VGA YUV422 (614 KB) does not.
- The ROM's "SHA-256 comparison failed … boot anyway" is benign: `esptool image-info hello.bin` reports checksum and validation hash **valid**, and `--flash-* keep` changes nothing. `[INFERENCE]` the ROM loader hashes only the segments it loads itself and skips the flash-mapped DROM/IROM ones, so a TinyGo image (ROM-loaded, with XIP segments) always shows this line. Not fixed; harmless.
- LED toggling not observed (no one watched the board); only the console output is claimed.

## 3. Restore

```
esptool --port /dev/cu.usbmodem101 -b 921600 --after no-reset write-flash --flash-mode keep --flash-freq keep --flash-size keep 0x0 backup/flash-1.bin
Flash will be erased from 0x00000000 to 0x007fffff...
Wrote 8388608 bytes (1013278 compressed) at 0x00000000 in 42.7 seconds (1570.0 kbit/s).
Hash of data verified.
esptool --port /dev/cu.usbmodem101 -b 921600 --before no-reset verify-flash --flash-mode keep --flash-freq keep --flash-size keep 0x0 backup/flash-1.bin
Verification successful (digest matched).
```

Boot after reset (`mon.py 8 --reset`, log `tmp/esp32cam/restore-boot.log`) is identical to the factory log from E0 (`factory-boot.log`): same loader segments (`load:0x3fce2820,len:0x118c` …, `entry 0x403c88b8`) and `Hello from Seeed Studio XIAO ESP32-S3 Sense` followed by dots. The board is left running the factory app. Expected drift: the app rewrites its NVS sector on boot (§1), so a later read differs from the backup only there.

## 4. Learnings (proposed, for Leader to file)

```
- 2026-11-09 [embedded] [high] esptool backups of a running ESP32 app differ in NVS if the app boots between reads; read twice with `--after no-reset` then `--before no-reset --after no-reset` (stub stays up) to get equal SHA-256 — ER-ESP32CAM E1
- 2026-11-09 [embedded] [medium] TinyGo 0.43 esp32s3 images boot with ROM "SHA-256 comparison failed … boot anyway" though esptool image-info says the hash is valid; benign — ER-ESP32CAM E1
- 2026-11-09 [embedded] [high] TinyGo 0.43 on ESP32-S3: USB Serial/JTAG console, I2C, LEDC work; LCD_CAM and GDMA are register structs only; no PSRAM (heap ≈ 385 KB internal) — ER-ESP32CAM E0/E1
```
