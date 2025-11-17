package graph

import (
	"github.com/itohio/EasyRobot/x/marshaller/storage"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// NewFileMap returns a factory function for creating file-based mapped storage.
// This is a convenience wrapper around storage.NewFileMap().
func NewFileMap() types.MappedStorageFactory {
	return storage.NewFileMap()
}

// NewMemoryMap returns a factory function for creating in-memory mapped storage.
// This is a convenience wrapper around storage.NewMemoryMap().
func NewMemoryMap() types.MappedStorageFactory {
	return storage.NewMemoryMap()
}

// NewTarMap exposes tar/tar.gz archive entries as read-only mapped storage.
// archivePath points to the tar file containing the logical graph files.
func NewTarMap(archivePath string) *storage.TarFactory {
	return storage.NewTarMap(archivePath)
}
