# Tar Storage Factory Update Plan

## Goal
Make `NewTarMap` transparent to callers while also supporting write workflows
via a `Commit()` method that rewrites the archive only when necessary.

## Requirements
1. `NewTarMap` must accept the archive filename when constructed and return a
   `TarFactory`.
2. The factory function will be invoked with the *logical* file path (e.g.
   `/tmp/nodes.graph`). It should:
   - Use the real file if it exists (preserving read/write semantics).
   - Otherwise, treat the request as a tar archive lookup using the base name
     (and normalized variants) of the path.
3. Archive-backed storage can be opened writable by materializing an on-disk
   temp file; these temps should be tracked so `Commit()` can rebuild the
   archive without keeping data in memory.
4. `Commit()` is a manual step that rewrites the archive (streaming) only when
   there are dirty entries, preserving untouched entries.
5. No code outside the factory needs to know whether storage came from tar or
   disk.
6. Tests must cover disk fallback, archive reads, writable temp handling, and
   commit behavior.

## Design
- Change `NewTarMap` signature to `func NewTarMap(archivePath string) *TarFactory`
  with a `Factory()` method that returns `types.MappedStorageFactory`.
- When the factory runs:
  - If `path` exists on disk, delegate to `newFileStorage`.
  - Else, derive candidate archive entry names from the path (normalized and
    base-name variants) and load from the tar (gzip-aware).
  - Keep existing mmap-based access for plain tar entries; gzip members are
    still streamed into temp files when needed for writes.
- `Commit()` streams the original archive into a temp file, skipping entries
  that have replacements, then appends all dirty temp entries before atomically
  renaming.
- Update docs/specs to explain the new semantics and usage pattern
  (`tf := storage.NewTarMap("/tmp/graph.tar"); factory := tf.Factory()`).

## Testing
- Update tar tests to:
  - Request entries via logical file names.
  - Verify disk fallback.
  - Exercise writable temps + `Commit()` rewriting (ensuring untouched entries
    remain).



