---
task: ER-ESP32CAM
owner: esp32cam
updated: 2026-11-09
---

# ER-ESP32CAM — milestone plan (esp32cam's working plan; the brief in `docs/tasks/` is Leader's)

## Hardware rules (owner, relayed by Leader 2026-11-09)

Owner: "i don't care about original firmware on the boards. no wifi required at all."

- No flash backups and no factory restores from now on. Test firmware stays on the board between sessions.
- No Wi-Fi in any code (also no `espradio` imports).
- Only `/dev/cu.usbmodem101` (MAC 1c:db:d4:74:55:80); flashing approved.
- PSRAM is not needed for the camera acceptance: QVGA YUV422 (153.6 KB) fits the ~385 KB internal heap.
- Superseded: the E1/E3 backup and restore steps. Their reports stay as the record; the backup images were deleted (Wi-Fi credentials) and only `SHA256SUMS` is kept.

## Milestones (step list from `docs/tasks/ER-CAM-steps.md`, Leader 2026-11-09)

Rule (owner): one step per round, one observable check written before starting, a report after every step (`ER-ESP32CAM-<step>.md`) even when it fails; a step that fails twice gets split smaller.

| Id | Milestone | State | Evidence |
|---|---|---|---|
| E0 | Identify board and sensor (read-only), TinyGo support, backend layout | done, verify accepted (`verification/ER-ESP32CAM-E01.md`); GDMA sentence corrected | `ER-ESP32CAM-E0.md` |
| E1 | Flash backup + TinyGo hello + restore | done, verify accepted (backup/restore part now obsolete; backups deleted by Leader) | `ER-ESP32CAM-E1.md` |
| E2 | Sensor ID over SCCB (OV3660, 0x3660) | done, verify accepted (`verification/ER-CAM-BATCH-1.md`); State section corrected | `ER-ESP32CAM-E2.md` |
| E3 | Frame over USB-serial, ERIM v1 + CRC, PNG of the room (acceptance) | done, awaiting verify | `ER-ESP32CAM-E3.md`, `ER-ESP32CAM-E3.png` |
| E3c | Fresh frames after the cameras moved; 'scaled' (full FOV, ISP scaler) vs 'windowed' (native crop, offset `o<x>,<y>;`) modes. Check: scaled FOV visibly wider than windowed | done (pass; scaled = full array, measured) | `ER-ESP32CAM-E3c.md`, `ER-ESP32CAM-E3c.png` |
| E3b | Move the probe into `drivers/camera` (`camera_esp32s3.go`, `sensor/ov3660`) on k210's interface (after K3); GDMA ring as a small reusable piece modelled on TinyGo's I2S driver, with a configurable channel (I2S uses ch0); switch to k210's `cmd/camframe`. Check: the same E3 capture from the driver package | blocked on k210's interface | — |
| E4 | GDMA ring continuous capture: double-buffered descriptor ring, no CPU copy on the capture path (in `_scratch` until E3b). Check: 100 consecutive frames, 0 drops, fps logged | done, verify accept with fix (`verification/ER-CAM-BATCH-2.md`; copy wording fixed in E4 §1) | `ER-ESP32CAM-E4.md`, `ER-ESP32CAM-E4.png` |
| E5 | Streaming: continuous QVGA to the host viewer over USB serial, every other frame (Leader's choice). Check: fps over 5 min, 0 CRC errors | done (pass: 1,625 images in 300 s, 0 CRC errors, 5.43 fps, MP4) | `ER-ESP32CAM-E5.md`, `ER-ESP32CAM-E5.png` |
| C1 | Sensor controls (OV3660): AE/manual exposure (µs), AGC/manual gain, AWB auto/freeze/manual, gamma toggle, live read-back; API proposal for k210 in the report. Check: manual exposure ×2 doubles a flat-patch mean (±15 %), read-back matches | done (pass: ratio 1.93; read-back exact to quantisation) | `ER-ESP32CAM-C1.md`, `ER-ESP32CAM-C1.png` |
| C2 | Matched comparison with the K210: same scene, 320×240 scaled, YUV422 → Y + colour PNG; (a) auto with read-backs logged, (b) same manual exposure, gain set for an equal flat-patch mean. Check: side-by-side PNGs + flat-patch σ, mean, edge sharpness in my evidence; shared table in k210's evidence | parked until k210's C1 (Leader) | — |
| M1 | SD card alive (read-only): list root + card size/free | **not passed**: card answers (6.13 GB SDHC, "SMISD") but every block reads as the same CID/CSD copy; no file system. Owner to check the card on the Mac | `ER-ESP32CAM-M1.md` |
| M2–M6 | Motion firmware (brief `docs/tasks/ER-ESP32CAM-MOTION.md`): M2 write `/MOTION/TEST.TXT`, M3 OV3660 JPEG, M4 motion detector, M5 combined 1 fps JPGs on motion, M6 robustness. SD rules: never format; only add files under `/MOTION/` | blocked on a readable card (M1) | — |
| E6 | PSRAM init: octal PSRAM up and mapped (TinyGo, ESP32-S3R8). Check: 8 MB write/read test pattern passes | planned (was "E4 octal PSRAM driver") | — |
| E7 | PSRAM frame buffers: GDMA into PSRAM with cache handling. Check: VGA (or the largest that fits) frame, CRC-clean, recognisable | planned | — |
| E8 | JPEG or higher-resolution streaming (sensor JPEG mode). Check: fps at SVGA/VGA | planned | — |
| later | NN on the chip (W9 kernels, ESP32-S3 SIMD) | later | — |
