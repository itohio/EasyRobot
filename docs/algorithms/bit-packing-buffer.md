---
type: algorithm
status: partial
tags: [easyrobot, buffer, bits]
workstream: [W2, W7]
module: "[[bit-buffer]]"
summary: "37 pass; vet stdmethods; Write/Read return bits"
source: [x/buffer/bit_buffer.go]
tests: [x/buffer/bit_buffer_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/buffer → ok (37 Bits tests pass, 2026-11-09); go vet ./x/buffer → stdmethods: WriteByte/ReadByte signatures; probe bufprobe: Write(3 bytes) returns n=24, io.Copy → 'invalid write result'"
---
# Bit-packing buffer `buffer.Bits`

**What it does:** packs values of `bitLength` (1–8) bits MSB-first into a growable byte slice and reads them back; used to carry k-bit symbols (e.g. 2-bit QPSK/4-FSK symbols) through byte-oriented `buffer.Reader/Writer` pipelines. Single shared cursor `pos` for reading *and* writing, plus `writtenBits` high-water mark.

**Where:** `x/buffer/bit_buffer.go` — `NewBits(bitLength, data...)` :42 (aliases caller bytes, no copy), `WriteBit` :119, `WriteByte(value)` :144-178, `Write` :183-189, `ReadBit` :193, `ReadByte() (byte,bool)` :210-236, `Read` :241-260, `SetPosition` :288, `Compact` :326, `ToBytes` :342 (masks padding), `ReadFromBytes` :367-401.

**Inputs → outputs:** values (low `bitLength` bits of each byte) ↔ packed bit stream.

## Completeness review

- Complete: yes for packing/unpacking.
- Tested: 37 tests pass (bit/byte I/O for widths 1–8, partial values, positioning, compaction, growth, interface compatibility).
- Correct: no — contract violations:
  - `Write` and `Read` return the number of **bits**, not elements (`:186`, `:259`). `io.Writer`/`buffer.Writer` require `0 ≤ n ≤ len(p)`; probe: `Write([3]byte)` → `n=24`, `io.Copy` into it fails with `invalid write result`.
  - `go vet` (stdmethods): `WriteByte(byte)` without `error` and `ReadByte() (byte, bool)` clash with `io.ByteWriter/ByteReader` (`:144`, `:210`). Baseline lists this as the package's vet failure.
  - Shared cursor: writing after a partial read overwrites unread data (probe: write AA BB CC, `Reset`, read AA, `WriteByte(11)` → `AA 11 CC`). `ReadFromBytes` resets `pos` to 0 (`:401`). `Len()` returns backing bytes (`:301`) whereas `Values.Len()` returns unread elements — inconsistent across `buffer.Reader` implementations.
  - Unaligned `ReadFromBytes` ORs into the first byte without clearing stale low bits (`:389`) [derived from code; not probed].

## TODOs

- [ ] Return element counts from `Write/Read` (bits via `Bits()`) — x/buffer/bit_buffer.go:183-189,241-260 #p1 #ER-023
- [ ] Rename `WriteByte/ReadByte` to `WriteValue/ReadValue` (or match io signatures) to clear vet — x/buffer/bit_buffer.go:144,210 #p2
- [ ] Separate read and write cursors (FIFO semantics) or document the shared cursor loudly — x/buffer/bit_buffer.go:14-28 #p2
- [ ] Make `Len()` mean unread values, consistent with `Values`/`Ring` — x/buffer/bit_buffer.go:301 #p2
- [ ] Mask the partial byte in unaligned `ReadFromBytes` — x/buffer/bit_buffer.go:384-395 #p3

## Links

- Module: [[bit-buffer]]
- Related: [[bit-ring-buffer]], [[modulation]]
