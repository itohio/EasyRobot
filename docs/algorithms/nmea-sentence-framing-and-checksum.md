---
type: algorithm
status: tested
tags: [easyrobot, devices, gps, protocol]
workstream: W1
module: "[[nmea]]"
summary: "pass (TestParseSentence, TestParser_ParseBytes, _Incomplete)"
source: [x/devices/nmea/parser.go]
tests: [x/devices/nmea/parser_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/devices/nmea → TestParseSentence (4 subtests), TestParser_ParseBytes, TestParser_ParseBytes_Incomplete pass (2026-11-09); probe tmp/easyrobot/devices/nmeaprobe"
---
# NMEA 0183 sentence framing and checksum

**What it does:** `ParseSentence` validates one `$…*HH` line (XOR checksum of the bytes between `$` and `*`) and splits it into talker+type and fields; `Parser.ParseBytes` accumulates a byte stream, cuts sentences at `$`…`\n`, drops invalid ones and returns the parsed list.

**Where:** `x/devices/nmea/parser.go:29-82` (`ParseSentence`), `:85-91` (`calculateChecksum`), `:94-161` (`Parser`, `ParseBytes`, `Reset`).

**Inputs → outputs:** `string` → `*Sentence{Type, Fields []string, Checksum, Raw, Timestamp}` or `ErrInvalidSentence`/`ErrChecksumMismatch`; `[]byte` chunks → `([]*Sentence, consumed int, error)`.

## Completeness review

- Complete: yes for `$` sentences with mandatory checksum. Not handled: `!` (AIS) sentences, sentences without checksum (allowed by NMEA for some sources), proprietary `$P…` are passed through as plain types.
- Tested: `TestParseSentence` checks type and non-empty fields for valid GGA/RMC and errors for bad checksum and missing `$`; `TestParser_ParseBytes` checks 2 sentences, types and `consumed == len(data)`; `_Incomplete` checks a sentence split across two calls — all pass (2026-11-09).
- Correct: yes for the tested scope. Caveats (read + probe):
  - `consumed` counts bytes of the **internal buffer**, not of the current call: split sentence → `c1=0, c2=67` while the second chunk was 42 bytes (`parser.go:111-155`). Harmless today (`Device` ignores it) but misleading API.
  - Unbounded buffer: if no `$` arrives (binary/UBX traffic on the same port) or a `$` is never followed by `\n`, `p.buf` grows forever (`parser.go:115-134`). Needs a max length.
  - Allocates per sentence (`string(...)`, `strings.Split`, `*Sentence`, `time.Now()`), fine on host, costly on MCU.

## TODOs

- [ ] Cap the parser buffer (drop to last `$` or clear when > N bytes) — x/devices/nmea/parser.go:115-134 #p2
- [ ] Return bytes consumed from the caller's chunk, or drop the return value — x/devices/nmea/parser.go:107-155 #p3

## Links

- Module: [[nmea]] · Field decoding: [[nmea-gga-rmc-decoding]] · Geodesy consumer: [[gps-geodesy]]
