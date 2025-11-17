# Storage Subpackage Specification

## Overview

The `x/marshaller/storage` package houses the concrete implementations of the
`types.MappedStorage` abstraction used by the graph marshaller. Each backend
exposes the same surface:

- `Map(offset, length)` – zero-copy byte access to underlying storage
- `Size()` – total logical size
- `Grow(size)` – extend writable stores
- `Close()` – release native resources
- (optional) `ReaderWriterSeeker()` – obtain an `io.ReadWriteSeeker`

## Supported Backends

| Backend | Build Tag | Characteristics |
|---------|-----------|-----------------|
| File (`file.go`) | `!tinygo` | Memory-mapped regular files via standard library `syscall` primitives. Read/write, growable. |
| Memory (`memory.go`) | none | Fully in-memory slice. Always writable, used in tests. |
| Flash (`flash.go`) | `tinygo` | Placeholder for MCU flash access (platform-specific). |
| Tar (`tar.go`) | `!tinygo` | Transparent access to disk files or tar/tar.gz entries with optional write+commit support. |
| Windows mmap helper (`mmap_segment_windows.go`) | `windows` | Uses `CreateFileMapping` / `MapViewOfFile` to back file/tar regions. |

All mmap operations must go through standard-library APIs (`syscall.Mmap`,
`CreateFileMapping`, etc.). Third-party packages are not permitted here to keep
portability and security guarantees.

## Tar/Tar.gz Storage

`NewTarMap(archivePath string)` now returns a `TarFactory` that can:

- expose a `types.MappedStorageFactory` via `Factory()` so existing callers can
  keep passing logical file paths (e.g. `/tmp/nodes.graph`);
- lazily create writable temp files when a path is modified and keep them off
  memory until `Commit()` is invoked;
- rebuild the archive (streaming, no in-memory buffering) on `Commit()`, merging
  untouched entries with the modified ones.

If the path exists on disk, it is memory-mapped directly. Otherwise, the factory
opens the configured archive and maps the entry matching the path (normalized
and base-name variants). The implementation has two modes:

- **Plain tar** – lazily maps the file region corresponding to the entry. Each
  `Map` call performs an aligned mmap of the requested sub-range.
- **Tar.gz** – streams the entry into an in-memory buffer one time, then serves
  subsequent `Map` calls from that buffer.

Writes and `Grow` are rejected to prevent corrupting archives. Attempts to open
without the separator or target a missing entry return descriptive errors (and
propagate `fs.ErrNotExist` for easy detection).

## Design Constraints

1. **Composition over inheritance** – each backend is a dedicated struct.
2. **Single Responsibility** – per backend file; helper utilities live in
   `mmap_*.go`.
3. **Return early** – guard invalid offsets/lengths before touching native APIs.
4. **Error handling** – wrap low-level errors with context (`fmt.Errorf("file storage: ...: %w", err)`).
5. **No naked returns** – explicit return values.
6. **Maximum function length** – keep critical methods ≤30 LOC; introduce
   helpers for complex logic (e.g., `_mapFileRegion`).

## Future Work

- Platform-specific flash implementations once TinyGo targets are finalized.
- Optional write-through tar backend that rewrites archives atomically.
- Helper CLI under `tools/` for inspecting mapped regions.


