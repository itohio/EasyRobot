---
type: algorithm
status: partial
tags: [easyrobot, marshaller, storage, tar]
workstream: [W1]
module: "[[marshallers]]"
summary: "6 tar tests pass"
source: [x/marshaller/storage/tar.go, x/marshaller/storage/buffer_storage.go]
tests: [x/marshaller/storage/tar_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/marshaller/storage → ok, 18 PASS incl. TestTarStorage* (2026-11-09)"
---
# Tar / tar.gz archive as mapped storage

**What it does:** `TarFactory` (`NewTarMap(archive)`) serves `MappedStorageFactory` paths from (1) an existing file on disk, else (2) an entry inside a plain tar (mmap of the entry's byte range) or tar.gz (entry decompressed into memory once). Writable opens create in-memory buffers; `Commit()` rewrites the archive (copy untouched entries + dirty buffers, temp file + rename).

**Where:** `NewTarMap` `tar.go:49-56`; `create` :62-92 (disk file wins, :71-75); read-only archive open :94-139; writable entry :141-163; `Commit` :181-275; entry copy :291-352; `tarStorage.Map` :387-437; `locateTarEntry` :658-691; gzip detection :592; name candidates :718-733; `bufferStorage` `buffer_storage.go:15-70`.

## Completeness review

- Complete: no.
  - SPEC says modified entries go to temp files "off memory until Commit" (`storage/SPEC.md` §Tar); code keeps them as in-memory `bufferStorage` (`tar.go:151-155`) — large graphs live fully in RAM.
  - If a logical path exists on disk, it is opened directly and the archive is bypassed even for writes (`tar.go:71-75`) — surprising when the caller expects archive semantics.
  - A new archive path ending in `.tar.gz` is written uncompressed: gzip is chosen only from the *existing* file (`currentArchiveIsGzip` :277-289, used at :205-217).
  - `writableEntry.openStorage` ignores `readOnly` (:174-179).
- Tested: plain tar map/read, tar.gz read, missing entry → `fs.ErrNotExist`, invalid path, plus write+commit cases in `tar_test.go` (all pass).
- Correct: unknown for archives > 8 GiB (classic tar header limits) and concurrent `Commit`.

## TODOs

- [ ] Spill dirty entries to temp files as the SPEC states — x/marshaller/storage/tar.go:151-155 #p3
- [ ] Choose compression from the archive extension for new archives — x/marshaller/storage/tar.go:205-217 #p3
- [ ] Make disk-file precedence explicit (option) — x/marshaller/storage/tar.go:71-75 #p3

## Links

- Module: [[marshallers]]; [[mapped-storage-file-memory]]; [[graph-marshaller-mmap-format]]
