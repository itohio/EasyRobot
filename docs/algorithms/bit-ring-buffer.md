---
type: algorithm
status: partial
tags: [easyrobot, buffer, bits]
workstream: [W2, W7]
module: "[[bit-buffer]]"
summary: "24 pass; WriteBit counts bits as values"
source: [x/buffer/bit_ring.go]
tests: [x/buffer/bit_ring_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/buffer → ok (24 BitRing tests pass, 2026-11-09); probe bufprobe: 4×WriteBit on 4-bit ring → Len 4 / Full (should be 1 value)"
---
# Bit ring buffer `buffer.BitRing`

**What it does:** fixed-capacity circular FIFO of `bitLength`-bit values (1–8) packed into bytes; separate head/tail bit positions. Same role as [[bit-packing-buffer]] but bounded and FIFO.

**Where:** `x/buffer/bit_ring.go` — `NewBitRing(bitLength, capacity)` :47, `WriteBit` :107-117, `WriteByte` :121-140, `Write` :145-153, `ReadBit` :157-167, `ReadByte` :171-190, `Read` :195-225, `Grow` :276-343 (repack into larger buffer), `Reset` :346.

**Inputs → outputs:** values ↔ bits, `len`/`capacity` counted in values.

## Completeness review

- Complete: yes.
- Tested: 24 tests pass (4/8-bit values, wraparound, partial reads, capacity, masking, grow, init validation, continuous write/read, bit I/O).
- Correct: no.
  - `WriteBit`/`ReadBit` change `len` by one **value** per bit (`:114`, `:164`): for `bitLength > 1` the value count and the full/empty checks go wrong (probe: four `WriteBit` on a 4-bit ring → `Len()=4`, `Full()=true`, though only one value was written). `TestBitRing_ReadBit_WriteBit` passes because it pairs bit writes with bit reads.
  - `Write/Read` return bit counts (`:150`, `:223`), same contract break as [[bit-packing-buffer]].
  - Per-bit `%` on every access (`writeBitAt/readBitAt` :227-246) — slow on MCUs; acceptable for low rates.

## TODOs

- [ ] Track a bit count (or forbid `WriteBit` when `bitLength > 1`) so `len`/full/empty stay in values — x/buffer/bit_ring.go:107-117,157-167 #p1 #ER-023
- [ ] Return element counts from `Write/Read` — x/buffer/bit_ring.go:145-153,195-225 #p1 #ER-023
- [ ] Byte-aligned fast path for `bitLength == 8` — x/buffer/bit_ring.go:121-190 #p3

## Links

- Module: [[bit-buffer]]
