---
type: algorithm
status: tested
tags: [easyrobot, devices, crsf, protocol]
workstream: [W6, W3]
module: "[[crsf-elrs]]"
summary: "pass (+ probe vs bitwise reference)"
source: [x/devices/crsf/channels.go, x/devices/crsf/encoder.go]
tests: [x/devices/crsf/crsf_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test → TestChannels* pass (2026-11-09); probe: pack/unpack == bit-by-bit LSB-first reference (4 patterns, 0 mismatches); all-992 payload e0 03 1f f8 c0 07 …; scaling matches (t-992)*5/8+1500 with truncation"
---
# CRSF RC channels packing (0x16) and µs scaling

**What it does:** packs 16 × 11-bit channel values into a 22-byte little-endian (LSB-first) bitstream and back; converts ticks ↔ servo µs with `us = (t−992)·5/8 + 1500`, `t = (us−1500)·8/5 + 992` (SPEC.md:129-130).

**Where:** `x/devices/crsf/channels.go:3-11` constants (`ChannelCenter`=992, `ChannelMax`=1984), `:19-25` `Microseconds`/`SetMicroseconds`, `:27-33` `ticksToMicroseconds`/`microsecondsToTicks`, `:36-40` `MarshalChannelsTo` (no alloc), `:43-47` `MarshalChannels`, `:50-57` `UnmarshalChannels`, `:60-75` `packChannels`, `:78-91` `unpackChannels`, `:94-107` `DecodeChannels`/`EncodeChannels`; zero-alloc frame path `x/devices/crsf/encoder.go:26-40` `WriteChannels`.

**Inputs → outputs:** `Channels{Ch [16]uint16}` (values masked to 11 bits on pack) ↔ `[]byte` len 22. µs helpers: `uint16` in/out, integer arithmetic, Go truncation toward zero.

## Completeness review

- Complete: yes. Pack handles 3-byte spans when `bitIdx > 5` (channels.go:69-71, :84-86); max index touched is 21.
- Tested: `TestChannelsPackUnpack` (172/992/1811/1500 round trip), `TestChannelsBoundary` (0, 1984, 0x7FF), `TestChannelsPackTo`, `TestChannelsMicroseconds` (1500/1000/2000 exact, 988/2012 ±1 µs round trip), `TestDecoderStream`. All round-trip only — no fixed wire vector. Pass (2026-11-09).
- Correct: yes. Probe (`tmp/easyrobot/devices/crsf`): `MarshalChannels` equals an independent bit-by-bit LSB-first packer and `UnmarshalChannels` inverts it for 4 patterns (0 mismatches); all-992 payload `e0 03 1f f8 c0 07 3e f0 81 0f 7c …` (hand-derived: 992=0x3E0 → e0, 03, 1f, f8 ✓); `WriteChannels` bytes == `WriteFrame(EncodeChannels)`. Scaling: 172→988, 191→1000, 992→1500, 1792→2000, 1811→2011 (exact 2011.875, truncated), 0→880, 1984→2120; 988 µs→173 ticks (exact 172.8).

**Defects / risks**
- `microsecondsToTicks` has no clamp: 500 µs → 64928 (uint16 wrap of −608), 3000 µs → 3392 (exceeds 11 bits, masked on pack to 1344) (channels.go:31-33).
- Truncation (not rounding) makes ticks→µs biased toward 1500 by up to 1 µs (1811 → 2011 instead of 2012).
- `Microseconds(ch)`/`SetMicroseconds(ch)` index without bounds check → panic for ch ≥ 16 (channels.go:19-25).
- `UnmarshalChannels` allocates a `*Channels` per frame (channels.go:54); `rx.Device` copies it again.

## TODOs

- [ ] Clamp `SetMicroseconds` to the 11-bit / 0..1984 tick domain — x/devices/crsf/channels.go:31 #p2
- [ ] Add fixed wire-vector test (all-992 payload / full frame `…7c ad`) — x/devices/crsf/crsf_test.go:106 #p2
- [ ] Round instead of truncate in µs conversion (matches OpenTX/ELRS 172→988, 1811→2012) — x/devices/crsf/channels.go:28 #p3
- [ ] Non-allocating `UnmarshalChannelsInto(dst *Channels, payload)` — x/devices/crsf/channels.go:50 #p3

## Links

- Module: [[crsf-elrs]] · Framing: [[crsf-frame-codec-and-crc8]] · Consumers: [[actuators]] (PWM/servo targets), [[apps]]
