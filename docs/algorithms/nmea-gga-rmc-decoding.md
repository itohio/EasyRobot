---
type: algorithm
status: partial
tags: [easyrobot, devices, gps, protocol]
workstream: W1
module: "[[nmea]]"
summary: "RMC 2.0 rejected, century bug, panic on bad coords, altitude wiped"
source: [x/devices/nmea/data.go, x/devices/nmea/device.go]
tests: [x/devices/nmea/parser_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 ./x/devices/nmea → TestParseGGA, TestMergeData pass; TestParseRMC FAIL 'nmea: invalid sentence' (2026-11-09); probe confirms year, panic and altitude defects"
---
# NMEA GGA/RMC field decoding and merge

**What it does:** Turns parsed GGA (fix) and RMC (recommended minimum) sentences into a `Data` struct (UTC time, lat/lon deg, altitude m, fix quality, satellites, HDOP, speed m/s, course deg) and merges successive updates; `Device` runs a serial read loop and calls `OnUpdate` with a copy.

**Where:** `x/devices/nmea/data.go:35-107` (`ParseGGA`), `:110-187` (`ParseRMC`), `:191-232` (`parseDMM`), `:235-263` (`MergeData`); `device.go:63-128` (`Device.Start`, `readLoop`, `processSentence`).

**Inputs → outputs:** `*Sentence` → `*Data` (float32 degrees; knots×0.514444 → m/s).

## Completeness review

- Complete: **no.** Talkers limited to `GP`/`GN` (`data.go:36,111`; `device.go:104-106`): GLONASS-only `GL`, BeiDou `GB/BD`, Galileo `GA` rejected (probe: `GLRMC` → invalid). No VTG/GSA/GSV/GLL/ZDA; GGA ignores fractional seconds and takes the date from the host clock (`data.go:52-57`); geoid separation (field 10) ignored.
- Tested: `TestParseGGA` checks lat 48.1173, lon 11.5167, fix 1, 8 sats, HDOP 0.9, alt 545.4 — pass. `TestMergeData` checks flags and copied fields — pass. `TestParseRMC` — **FAIL**.
- Correct: **no.**
  1. **RMC rejects NMEA 2.0 sentences** — `data.go:115` requires ≥ 12 fields, but the classic 2.0 RMC (the test's sentence) has 11; 12 appears only with the 2.3 mode indicator. This is the cause of the `TestParseRMC` failure. Probe: 11 fields → error; 12/13 fields → lat/lon/speed 11.52 m/s/course right. Fix: require ≥ 9 (through date).
  2. **Year off by a century for pre-2000 dates** — `data.go:177-181` adds 2000 then the `< 2000` branch can never trigger: `230394` → 2094-03-23 (probe). GPS week rollover makes "YY < 80 ⇒ 20YY else 19YY" the usual rule.
  3. **Panic on malformed coordinates** — `parseDMM` slices `dmmStr[:dotIdx-2]` without checking `dotIdx ≥ 2` (`data.go:210`): lat `.5` → `slice bounds out of range [:-2]` (probe). A corrupted-but-checksum-valid or hostile sentence crashes the read goroutine.
  4. **RMC wipes altitude** — `MergeData` copies `Altitude` whenever `PosValid` (`data.go:240-245`); RMC has no altitude, so GGA 545.4 → RMC → 0.0 (probe).
  - Also: GGA sets `PosValid` when only one of lat/lon parsed (`:62-77`); RMC status `V` leaves stale validity semantics to the merge; S/W hemispheres correct (probe: 33.8667 S, 151.2 W → negative).

## TODOs

- [ ] Accept 11-field (NMEA 2.0) RMC — x/devices/nmea/data.go:115 #p1 #ER-025
- [ ] Fix century handling for RMC dates — x/devices/nmea/data.go:177-181 #p1 #ER-025
- [ ] Guard `parseDMM` against `dotIdx < 2` / short strings — x/devices/nmea/data.go:205-213 #p1 #ER-025
- [ ] Only merge altitude from GGA (separate validity flags per field group) — x/devices/nmea/data.go:240-245 #p1 #ER-025
- [ ] Accept any talker ID (match on the last 3 chars of the type) — x/devices/nmea/data.go:36,111; device.go:104-106 #p2
- [ ] Add tests for 2.3 RMC, S/W hemispheres, malformed fields, merge order — x/devices/nmea/parser_test.go #p2

## Links

- Module: [[nmea]] · Framing: [[nmea-sentence-framing-and-checksum]] · Geodesy: [[gps-geodesy]]
