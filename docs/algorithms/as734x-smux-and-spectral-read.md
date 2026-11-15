---
type: algorithm
status: unverified
tags: [easyrobot, devices, sensors, spectral, i2c]
workstream: W1
module: "[[sensors]]"
summary: "unverified (integration-time overflow found)"
source: [x/devices/as734x/device.go, x/devices/as734x/as7341.go, x/devices/as734x/as7343.go, x/devices/as734x/bank.go, x/devices/as734x/bus.go, x/devices/as734x/smux.go, x/devices/as734x/registers.go, x/devices/as734x/types.go]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "no test files; build/vet/linux-arm64 cross ok (2026-11-09); probe on verbatim copy: integrationTimeUs(29,599)=50040 µs ok, integrationTimeUs(255,65535)=3690987 (true 46640250) → uint32 overflow. No hardware."
---
# AS7341/AS7343 variant detection, SMUX configuration and spectral read

**What it does:** detects AS7341 (10 ch) vs AS7343 (18 ch) over I²C, programs ATIME/ASTEP/AGAIN, and reads raw channel counts. AS7341: two SMUX passes (F1–F4+Clear+NIR, then F5–F8+Clear+NIR) merged into 10 values; AS7343: auto-SMUX 18-channel mode, one 36-byte read. AS7343 register bank switching is handled transparently.

**Where:**
- `x/devices/as734x/device.go:47-72` `Configure`; `:96-98` `powerOn`; `:100-125` `detectVariant`/`readAS7343ID`; `:131-136` `integrationTimeUs`; `:138-153` `waitForDataReady` (5 ms poll); `:155-166` `enableSpectral`.
- `x/devices/as734x/bus.go:3-63` bank-aware `readReg/writeReg/readRegs/writeRegs` and raw I²C helpers.
- `x/devices/as734x/bank.go:3-45` `setRegisterBank` (reg ≥ 0x80 → bank 0, else bank 1), `setBankAS7343`, `forceBankAS7343` (CFG0 0xBF bit 4).
- `x/devices/as734x/smux.go:4-23` SMUX tables (low, high, flicker).
- `x/devices/as734x/as7341.go:8-23` `configureAS7341`; `:34-77` `setSmuxCommand` (CFG6 0xAF bits 4:3), `enableSmux` (ENABLE bit 4, poll ≤50×2 ms), `loadSmux` (write 20 B to RAM 0x00); `:79-141` `readAS7341`/`captureBlock`/`readChannelBlock` (12 B from 0x95, little-endian); `:143-156` `mergeAS7341Channels` (clear/NIR averaged across passes).
- `x/devices/as734x/as7343.go:5-26` `configureAS7343`; `:33-41` `setAutoSmuxAS7343` (CFG20 0xD6 bits 6:5 = 3); `:56-90` `readAS7343` (36 B from 0x95); `:82-90` `decodeAS7343Channels`; `:92-98` `as7343Saturation`.
- `x/devices/as734x/types.go` `Config`, `DefaultConfig` (`:91-99`: 16×, ATIME 29, ASTEP 599, flicker on), `Gain.Multiplier`, `RawMeasurement`.

**Inputs → outputs:** `Configure(Config) error`; `Read() (RawMeasurement, error)` → `Channels []uint16` (AS7341 order F1..F8, Clear, NIR; AS7343 auto-SMUX order), `IntegrationUs`, `Gain`, `Saturated`, `Flicker`, `Timestamp`. Integration time t = (ATIME+1)(ASTEP+1)·2.78 µs; default 50.04 ms (probe). No basic-count normalisation (counts / (gain·t)) — left to callers.

**Register sequence vs datasheet** (all datasheet values recalled from memory and the Adafruit AS7341 library, not fetched — [INFERENCE]):
- AS7341 ID 0x92 bits 7:2 = 0x09 → mask `0xFC == 0x24` (device.go:102) matches. CFG1 0xAA AGAIN 0..10 (0.5×..512×), ASTEP 0xCA/0xCB, CFG6 SMUX_CMD write=2, SMUX RAM 0x00–0x13, SMUXEN self-clearing, STATUS2 0xA3 AVALID bit 6, CH0 data at 0x95 — consistent.
- SMUX tables `smuxLowConfig`/`smuxHighConfig`/`smuxFlickerConfig` match Adafruit `setup_F1F4_Clear_NIR`, `setup_F5F8_Clear_NIR`, `FDConfig` byte-for-byte as recalled.
- AS7343: ID 0x5A = 0x81 in bank 1, CFG0 0xBF REG_BANK bit 4, ASTEP 0xD4/0xD5, CFG1 0xC6 AGAIN ≤12 (2048×), CFG20 0xD6 auto_smux=3 (18 ch), STATUS2 0x90, DATA_0 0x95 — consistent as recalled.

## Completeness review

- Complete: no — no power-down/close, no ASTATUS (0x94) latch read before data [INFERENCE: datasheet recommends reading ASTATUS to latch a consistent data set], AS7341 saturation ignores analog flag, no wait-time/SP_EN interrupt use, calibration/reconstruction API in `calib.md`/`SPD_GUIDE.md` (e.g. `MeasureDarkCurrent`, `ReconstructSpectrum`) is documentation only — no such Go symbols exist.
- Tested: no test files (`go test ./x/devices/as734x` → `[no test files]`, 2026-11-09). Probe on a verbatim copy of `integrationTimeUs`: (29,599) → 50040 µs (correct); (255,65535) → 3 690 987 µs, true value 46 640 250 µs → overflow.
- Correct: unknown (no hardware); one confirmed arithmetic defect.

**Defects / risks**
- `integrationTimeUs` overflows uint32 when (ATIME+1)(ASTEP+1) > 1 544 925 (device.go:135; probe) → wrong `IntegrationUs` for long integrations; compute in uint64 or `steps*delta*278/100`.
- AS7343 bank state desync: `readAS7343ID` calls `forceBankAS7343(1)` (device.go:121) without updating `d.bank` (still 0), so after detection the chip sits in bank 1 while the driver believes bank 0; the first accesses to ≥0x80 registers in `configureAS7343` skip the switch (bank.go:14-16) [INFERENCE: on AS7343, registers ≥0x80 are only reachable with REG_BANK=0] → ATIME/ASTEP/AGAIN writes likely land in the wrong bank.
- `powerOn` writes ENABLE before variant detection (device.go:53, 97); on an AS7343 left in bank 1 that write is lost [INFERENCE].
- AS7341 `Saturated` checks only bit 4 (digital) (as7341.go:90); AS7343 checks 0x10|0x08 (as7343.go:97).
- Default `Read()` on AS7341 costs 2× (SMUX load + ~50 ms integration + 5 ms poll granularity) plus 200 ms flicker sleep because `DefaultConfig` enables flicker (types.go:97, as7341.go:89, 200) → ≈300 ms/blocking call; the flicker pass also toggles ENABLE to 0 every read (as7341.go:181).
- Per-read heap allocations: as7341.go:125 (12 B), :144 (10×u16), as7343.go:63 (36 B), :83 (18×u16), bus.go:59 (`writeRegsRaw` per SMUX load).
- No mutex; not safe for concurrent `Read`.

## TODOs

- [ ] Fix `integrationTimeUs` uint32 overflow — x/devices/as734x/device.go:135 #p1 #backlog
- [ ] Set `d.bank = 1` after `forceBankAS7343(1)` in detection (or restore bank 0) — x/devices/as734x/device.go:121 #p1 #backlog
- [ ] Include analog saturation (STATUS2 bit 3) for AS7341 — x/devices/as734x/as7341.go:90 #p2
- [ ] Unit tests with a fake I²C bus: variant detection, bank switching, SMUX write sequence, channel merge/decode, integration time — x/devices/as734x/device.go:100 #p2
- [ ] Default `FlickerEnable` to false or decouple flicker from `Read` — x/devices/as734x/types.go:97 #p2
- [ ] Implement `devices.Spectrometer` (or document why not) and either implement or mark `calib.md`/`SPD_GUIDE.md` APIs as proposals — x/devices/as734x/calib.md:184 #p3
- [ ] Reuse read buffers to avoid per-read allocation (TinyGo) — x/devices/as734x/as7341.go:125 #p3

## Links

- Module: [[sensors]] · Related: [[as734x-flicker-detection]] · Downstream: [[colorscience]] · Interfaces: [[device-io]]
