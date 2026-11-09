---
type: algorithm
status: partial
tags: [easyrobot, marshaller, storage, mmap]
workstream: [W1, W6]
module: "[[marshallers]]"
summary: "18 pass (pkg)"
source: [x/marshaller/storage/file.go, x/marshaller/storage/memory.go, x/marshaller/storage/flash.go, x/marshaller/storage/mmap_segment_unix.go, x/marshaller/storage/mmap_segment_windows.go, x/marshaller/storage/mmap_segment_stub.go, x/marshaller/types/types.go]
tests: [x/marshaller/storage/file_test.go, x/marshaller/storage/memory_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/marshaller/storage → ok, 18 PASS (2026-11-09); graphprobe shows NewMemoryMap cannot round-trip (2026-11-09)"
---
# Mapped storage back ends: mmap file, memory, flash

**What it does:** implements `types.MappedStorage` (`Map(offset,len) → MappedRegion{Bytes,Sync,Unmap}`, `Size`, `Grow`, `Close`, optional `ReaderWriterSeeker`) over (a) `syscall.Mmap` / `MapViewOfFile` file segments, (b) a Go byte slice, (c) a TinyGo flash byte slice. The graph marshaller is the only consumer.

**Where:** interfaces `x/marshaller/types/types.go:257-300`; file: `fileStorage.Map` `file.go:83-149` (stat-refresh each call), `Grow` :168-213 (writes zeros), `Close` :215-240, `NewFileMap` :299-315, page-aligned segment mapping `mmap_segment_unix.go:12-64`; memory: `memory.go:35-231`, `NewMemoryMap` :224-230; flash (`//go:build tinygo`): `flash.go:45-303`, `NewFlashMap` :267-284 (always errors), `NewFlashMapFromData` :286-292.

## Completeness review

- Complete: no.
  - `NewMemoryMap()` ignores `path` and returns a new empty storage on every call (`memory.go:224-230`); `memoryStorage.Close` drops the buffer (:111-116). A marshal→unmarshal through the same factory therefore reads empty storage (graphprobe: `region extends beyond storage … size: 0`).
  - `memoryStorage.Grow` reallocates (`memory.go:87-108`): regions mapped before `Grow` keep pointing at the old slice; writes through them are silently lost.
  - `NewFlashMapFromData` returns the **same** byte slice for every path (`flash.go:286-292`), so node, edge and data files of a graph overlap on MCU flash. `NewFlashMap` is a placeholder that always errors (:267-284). No erase/page-write logic.
  - `fileStorage.Grow` writes the whole extension as one zero buffer (`file.go:202-203`) — O(size) allocation; `os.File.Truncate` would avoid it. Never shrinks: rewriting a smaller graph leaves stale tail bytes (harmless because headers carry counts).
  - `fileStorage.Close` leaves regions mapped (by design, `file.go:224-226`); unmapping is the caller's job.
- Tested: 18 PASS (storage pkg incl. tar): file map/unmap/write-read/sync, read-only rejection, bad offsets, memory map/grow errors, factories.
- Correct: unknown for concurrent use; no test maps, grows, then writes through the old region.

## TODOs

- [ ] Path-keyed in-memory factory that survives `Close` (needed for tests and RAM-only MCUs) — x/marshaller/storage/memory.go:224-230 #p2
- [ ] Flash factory: one region per path (partition table) — x/marshaller/storage/flash.go:286-292 #p1 #ER-027
- [ ] Document or prevent stale regions after `Grow` — x/marshaller/storage/memory.go:87-108 #p3
- [ ] Use `Truncate` in `Grow` — x/marshaller/storage/file.go:202-203 #p3

## Links

- Module: [[marshallers]]; consumer [[graph-marshaller-mmap-format]]; archive back end [[tar-archive-mapped-storage]]; MCU targets [[build-targets]]
