//go:build !tinygo

package storage

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sync"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// fileRegion implements MappedRegion for file-based storage
type fileRegion struct {
	segment  *mappedSegment
	readOnly bool
}

func newFileRegion(file *os.File, offset, length int64, readOnly bool) (*fileRegion, error) {
	if length <= 0 {
		return nil, fmt.Errorf("file storage: invalid map length %d", length)
	}
	if length > int64(math.MaxInt) {
		return nil, fmt.Errorf("file storage: region too large (%d bytes)", length)
	}

	seg, err := mapFileSegment(file, offset, int(length), readOnly)
	if err != nil {
		return nil, fmt.Errorf("file storage: failed to map region: %w", err)
	}

	return &fileRegion{
		segment:  seg,
		readOnly: readOnly,
	}, nil
}

func (r *fileRegion) Bytes() []byte {
	return r.segment.view
}

func (r *fileRegion) Size() int64 {
	return int64(len(r.segment.view))
}

func (r *fileRegion) Sync() error {
	if r.readOnly {
		return nil
	}
	return syncSegment(r.segment)
}

func (r *fileRegion) Unmap() error {
	return unmapSegment(r.segment)
}

// fileStorage implements MappedStorage for file-based storage
type fileStorage struct {
	mu       sync.RWMutex
	file     *os.File
	readOnly bool
	size     int64
	regions  map[*fileRegion]struct{} // Track mapped regions for cleanup
}

func newFileStorage(file *os.File, readOnly bool) (*fileStorage, error) {
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("file storage: failed to stat file: %w", err)
	}

	return &fileStorage{
		file:     file,
		readOnly: readOnly,
		size:     info.Size(),
		regions:  make(map[*fileRegion]struct{}),
	}, nil
}

func (s *fileStorage) Map(offset, length int64) (types.MappedRegion, error) {
	s.mu.RLock()
	if s.file == nil {
		s.mu.RUnlock()
		return nil, errors.New("file storage: storage is closed")
	}

	// Refresh size from file to ensure we have current size before mapping
	// This is important to avoid stale size issues after file growth
	info, err := s.file.Stat()
	if err != nil {
		s.mu.RUnlock()
		return nil, fmt.Errorf("file storage: failed to stat file for size refresh: %w", err)
	}
	currentSize := info.Size()
	s.mu.RUnlock()

	// Validate offset and length using the fresh size (no lock needed for read-only validation)
	if offset < 0 || offset > currentSize {
		return nil, fmt.Errorf("file storage: invalid offset %d (size: %d)", offset, currentSize)
	}

	if length == 0 {
		length = currentSize - offset
	}

	if offset+length > currentSize {
		return nil, fmt.Errorf("file storage: region extends beyond file (offset: %d, length: %d, size: %d)", offset, length, currentSize)
	}

	if length <= 0 {
		return nil, errors.New("file storage: cannot map empty region")
	}

	// Check if storage is closed and capture file reference
	s.mu.RLock()
	if s.file == nil {
		s.mu.RUnlock()
		return nil, errors.New("file storage: storage is closed")
	}
	fileToMap := s.file
	readOnlyToMap := s.readOnly
	s.mu.RUnlock()

	// Create region while NOT holding lock to avoid blocking syscalls
	// CreateFileMapping creates a separate handle, so closing the storage's
	// file handle won't invalidate an already-created region
	region, err := newFileRegion(fileToMap, offset, length, readOnlyToMap)
	if err != nil {
		return nil, err
	}

	// Acquire write lock to add region to map and verify storage is still open
	s.mu.Lock()
	if s.file == nil || s.regions == nil {
		// Storage was closed while we were creating the region
		s.mu.Unlock()
		_ = region.Unmap() // Clean up region we just created
		return nil, errors.New("file storage: storage is closed")
	}

	// Add to regions map atomically
	s.regions[region] = struct{}{}
	s.mu.Unlock()

	return region, nil
}

func (s *fileStorage) Size() (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.file == nil {
		return 0, errors.New("file storage: storage is closed")
	}

	// Refresh size from file
	info, err := s.file.Stat()
	if err != nil {
		return 0, fmt.Errorf("file storage: failed to stat file: %w", err)
	}
	s.size = info.Size()
	return s.size, nil
}

func (s *fileStorage) Grow(size int64) error {
	if s.readOnly {
		return errors.New("file storage: cannot grow read-only storage")
	}

	if size < 0 {
		return fmt.Errorf("file storage: invalid size %d", size)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.file == nil {
		return errors.New("file storage: storage is closed")
	}

	// Refresh size from file to get current size (file might have grown externally)
	info, err := s.file.Stat()
	if err != nil {
		return fmt.Errorf("file storage: failed to stat file: %w", err)
	}
	currentSize := info.Size()
	s.size = currentSize

	if size <= currentSize {
		return nil // Already large enough
	}

	// Extend file by writing zeros at the end
	if _, err := s.file.Seek(currentSize, 0); err != nil {
		return fmt.Errorf("file storage: failed to seek: %w", err)
	}

	// Write zeros to extend file
	zeros := make([]byte, size-currentSize)
	if _, err := s.file.Write(zeros); err != nil {
		return fmt.Errorf("file storage: failed to grow file: %w", err)
	}

	if err := s.file.Sync(); err != nil {
		return fmt.Errorf("file storage: failed to sync file: %w", err)
	}

	s.size = size
	return nil
}

func (s *fileStorage) Close() error {
	s.mu.Lock()
	if s.file == nil {
		s.mu.Unlock()
		return nil // Already closed
	}

	// Mark storage as closed by setting file to nil
	// This prevents new regions from being created
	fileToClose := s.file
	s.file = nil
	// Don't clear regions map - regions will unmap themselves when their callers call Unmap()
	// This prevents unmapping regions that are currently in use
	s.mu.Unlock()

	// Close the file handle
	// On Windows, file handles can be closed even if mappings exist - the mappings
	// remain valid until they are explicitly unmapped. This is safe because:
	// 1. CreateFileMapping creates a separate handle that keeps the file open
	// 2. Closing the file handle doesn't invalidate existing mappings
	// 3. Regions will unmap themselves when their callers are done
	err := fileToClose.Close()

	return err
}

// ReaderWriterSeeker returns a view that implements io.Reader, io.Writer, and io.Seeker
// This makes fileStorage implement ReaderWriterSeekerStorage interface
func (s *fileStorage) ReaderWriterSeeker() (io.ReadWriteSeeker, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.file == nil {
		return nil, errors.New("file storage: storage is closed")
	}

	return &fileView{file: s.file, readOnly: s.readOnly}, nil
}

// fileView implements io.ReadWriteSeeker for file storage
type fileView struct {
	file     *os.File
	readOnly bool
	mu       sync.Mutex
}

func (v *fileView) Read(p []byte) (n int, err error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.file == nil {
		return 0, io.EOF
	}

	return v.file.Read(p)
}

func (v *fileView) Write(p []byte) (n int, err error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.file == nil {
		return 0, fmt.Errorf("file storage: storage is closed")
	}

	if v.readOnly {
		return 0, fmt.Errorf("file storage: cannot write to read-only storage")
	}

	return v.file.Write(p)
}

func (v *fileView) Seek(offset int64, whence int) (int64, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.file == nil {
		return 0, fmt.Errorf("file storage: storage is closed")
	}

	return v.file.Seek(offset, whence)
}

// NewFileMap returns a factory function for creating file-based mapped storage
func NewFileMap() types.MappedStorageFactory {
	return func(path string, readOnly bool) (types.MappedStorage, error) {
		var flags int
		if readOnly {
			flags = os.O_RDONLY
		} else {
			flags = os.O_RDWR | os.O_CREATE
		}

		file, err := os.OpenFile(path, flags, 0644)
		if err != nil {
			return nil, fmt.Errorf("file map: failed to open file %s: %w", path, err)
		}

		return newFileStorage(file, readOnly)
	}
}
