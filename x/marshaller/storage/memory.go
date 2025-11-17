package storage

import (
	"fmt"
	"io"
	"sync"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// memoryRegion implements MappedRegion for in-memory storage
type memoryRegion struct {
	data []byte
}

func (r *memoryRegion) Bytes() []byte {
	return r.data
}

func (r *memoryRegion) Size() int64 {
	return int64(len(r.data))
}

func (r *memoryRegion) Sync() error {
	// No-op for in-memory storage
	return nil
}

func (r *memoryRegion) Unmap() error {
	// No-op for in-memory storage
	return nil
}

// memoryStorage implements MappedStorage for in-memory storage
type memoryStorage struct {
	mu   sync.RWMutex
	data []byte
}

func newMemoryStorage(initialSize int64) *memoryStorage {
	if initialSize < 0 {
		initialSize = 0
	}
	return &memoryStorage{
		data: make([]byte, initialSize),
	}
}

func (s *memoryStorage) Map(offset, length int64) (types.MappedRegion, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.data == nil {
		return nil, fmt.Errorf("memory storage: storage is closed")
	}

	size := int64(len(s.data))
	if offset < 0 || offset > size {
		return nil, fmt.Errorf("memory storage: invalid offset %d (size: %d)", offset, size)
	}

	if length == 0 {
		length = size - offset
	}

	if offset+length > size {
		return nil, fmt.Errorf("memory storage: region extends beyond storage (offset: %d, length: %d, size: %d)", offset, length, size)
	}

	// Return a slice view of the data
	region := &memoryRegion{
		data: s.data[offset : offset+length],
	}
	return region, nil
}

func (s *memoryStorage) Size() (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.data == nil {
		return 0, fmt.Errorf("memory storage: storage is closed")
	}
	return int64(len(s.data)), nil
}

func (s *memoryStorage) Grow(size int64) error {
	if size < 0 {
		return fmt.Errorf("memory storage: invalid size %d", size)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data == nil {
		return fmt.Errorf("memory storage: storage is closed")
	}

	currentSize := int64(len(s.data))
	if size <= currentSize {
		return nil // Already large enough
	}

	// Grow the slice
	newData := make([]byte, size)
	copy(newData, s.data)
	s.data = newData
	return nil
}

func (s *memoryStorage) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = nil
	return nil
}

// ReaderWriterSeeker returns a view that implements io.Reader, io.Writer, and io.Seeker
// This makes memoryStorage implement ReaderWriterSeekerStorage interface
func (s *memoryStorage) ReaderWriterSeeker() (io.ReadWriteSeeker, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.data == nil {
		return nil, fmt.Errorf("memory storage: storage is closed")
	}

	return &memoryView{storage: s, pos: 0}, nil
}

// memoryView implements io.ReadWriteSeeker for memory storage
type memoryView struct {
	storage *memoryStorage
	pos     int64
	mu      sync.RWMutex
}

func (v *memoryView) Read(p []byte) (n int, err error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	v.storage.mu.RLock()
	defer v.storage.mu.RUnlock()

	if v.storage.data == nil {
		return 0, io.EOF
	}

	size := int64(len(v.storage.data))
	if v.pos >= size {
		return 0, io.EOF
	}

	available := size - v.pos
	toRead := int64(len(p))
	if toRead > available {
		toRead = available
	}

	copy(p, v.storage.data[v.pos:v.pos+toRead])
	v.pos += toRead
	return int(toRead), nil
}

func (v *memoryView) Write(p []byte) (n int, err error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.storage.mu.Lock()
	defer v.storage.mu.Unlock()

	if v.storage.data == nil {
		return 0, fmt.Errorf("memory storage: storage is closed")
	}

	size := int64(len(v.storage.data))
	needed := v.pos + int64(len(p))
	if needed > size {
		// Grow storage
		newData := make([]byte, needed)
		copy(newData, v.storage.data)
		v.storage.data = newData
	}

	copy(v.storage.data[v.pos:], p)
	v.pos += int64(len(p))
	return len(p), nil
}

func (v *memoryView) Seek(offset int64, whence int) (int64, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.storage.mu.RLock()
	defer v.storage.mu.RUnlock()

	if v.storage.data == nil {
		return 0, fmt.Errorf("memory storage: storage is closed")
	}

	size := int64(len(v.storage.data))
	var newPos int64

	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = v.pos + offset
	case io.SeekEnd:
		newPos = size + offset
	default:
		return 0, fmt.Errorf("memory storage: invalid whence: %d", whence)
	}

	if newPos < 0 {
		return 0, fmt.Errorf("memory storage: negative position: %d", newPos)
	}

	v.pos = newPos
	return v.pos, nil
}

// NewMemoryMap returns a factory function for creating in-memory mapped storage
func NewMemoryMap() types.MappedStorageFactory {
	return func(path string, readOnly bool) (types.MappedStorage, error) {
		// For in-memory storage, path is ignored
		// Initial size is 0, will grow as needed
		return newMemoryStorage(0), nil
	}
}

