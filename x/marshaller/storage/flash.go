//go:build tinygo

package storage

import (
	"errors"
	"fmt"
	"io"
	"sync"
	"unsafe"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// flashRegion implements MappedRegion for flash-based storage
// Flash memory is typically read-only or has limited write cycles
type flashRegion struct {
	data   []byte
	offset int64
}

func (r *flashRegion) Bytes() []byte {
	return r.data
}

func (r *flashRegion) Size() int64 {
	return int64(len(r.data))
}

func (r *flashRegion) Sync() error {
	// For flash, sync may require special handling
	// Some microcontrollers support flash writes, but they're slow
	// This is a placeholder - actual implementation depends on platform
	return nil
}

func (r *flashRegion) Unmap() error {
	// No-op for flash - data is directly accessible
	return nil
}

// flashStorage implements MappedStorage for flash-based storage
// This is a basic implementation that assumes flash is accessible as a byte slice
// Platform-specific implementations may need to use different approaches
type flashStorage struct {
	mu       sync.RWMutex
	data     []byte
	readOnly bool
	size     int64
}

// NewFlashStorage creates a flash storage from a byte slice
// This assumes the flash memory is already mapped/accessible
// For actual TinyGo implementations, you may need to:
// 1. Use platform-specific flash APIs (e.g., machine.FlashData)
// 2. Map flash memory regions using linker scripts
// 3. Use device-specific flash access methods
func newFlashStorage(data []byte, readOnly bool) *flashStorage {
	return &flashStorage{
		data:     data,
		readOnly: readOnly,
		size:     int64(len(data)),
	}
}

func (s *flashStorage) Map(offset, length int64) (types.MappedRegion, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.data == nil {
		return nil, errors.New("flash storage: storage is closed")
	}

	if offset < 0 || offset > s.size {
		return nil, fmt.Errorf("flash storage: invalid offset %d (size: %d)", offset, s.size)
	}

	if length == 0 {
		length = s.size - offset
	}

	if offset+length > s.size {
		return nil, fmt.Errorf("flash storage: region extends beyond storage (offset: %d, length: %d, size: %d)", offset, length, s.size)
	}

	// Return a slice view of the flash data
	// Note: This is safe because flash data is typically read-only or has special write handling
	region := &flashRegion{
		data:   s.data[offset : offset+length],
		offset: offset,
	}
	return region, nil
}

func (s *flashStorage) Size() (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.size, nil
}

func (s *flashStorage) Grow(size int64) error {
	if s.readOnly {
		return errors.New("flash storage: cannot grow read-only storage")
	}

	if size < 0 {
		return fmt.Errorf("flash storage: invalid size %d", size)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data == nil {
		return errors.New("flash storage: storage is closed")
	}

	if size <= s.size {
		return nil // Already large enough
	}

	// Flash growth is typically not supported or requires special handling
	// This would need platform-specific implementation
	// For now, we'll just extend the slice if possible (in-memory simulation)
	// In real implementations, this would need to use flash erase/write APIs
	newData := make([]byte, size)
	copy(newData, s.data)
	s.data = newData
	s.size = size

	return nil
}

func (s *flashStorage) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = nil
	return nil
}

// ReaderWriterSeeker returns a view that implements io.Reader, io.Writer, and io.Seeker
// This makes flashStorage implement ReaderWriterSeekerStorage interface
func (s *flashStorage) ReaderWriterSeeker() (io.ReadWriteSeeker, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.data == nil {
		return nil, errors.New("flash storage: storage is closed")
	}

	return &flashView{storage: s, pos: 0, readOnly: s.readOnly}, nil
}

// flashView implements io.ReadWriteSeeker for flash storage
type flashView struct {
	storage  *flashStorage
	pos      int64
	readOnly bool
	mu       sync.Mutex
}

func (v *flashView) Read(p []byte) (n int, err error) {
	v.mu.Lock()
	defer v.mu.Unlock()

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

func (v *flashView) Write(p []byte) (n int, err error) {
	if v.readOnly {
		return 0, fmt.Errorf("flash storage: cannot write to read-only storage")
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	v.storage.mu.Lock()
	defer v.storage.mu.Unlock()

	if v.storage.data == nil {
		return 0, fmt.Errorf("flash storage: storage is closed")
	}

	// Flash writes are typically slow and require special handling
	// This is a placeholder - real implementation would use flash write APIs
	size := int64(len(v.storage.data))
	needed := v.pos + int64(len(p))
	if needed > size {
		// Flash growth typically not supported
		return 0, fmt.Errorf("flash storage: cannot grow flash storage")
	}

	copy(v.storage.data[v.pos:], p)
	v.pos += int64(len(p))
	return len(p), nil
}

func (v *flashView) Seek(offset int64, whence int) (int64, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.storage.mu.RLock()
	defer v.storage.mu.RUnlock()

	if v.storage.data == nil {
		return 0, fmt.Errorf("flash storage: storage is closed")
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
		return 0, fmt.Errorf("flash storage: invalid whence: %d", whence)
	}

	if newPos < 0 {
		return 0, fmt.Errorf("flash storage: negative position: %d", newPos)
	}

	v.pos = newPos
	return v.pos, nil
}

// NewFlashMap returns a factory function for creating flash-based mapped storage
// This is a basic implementation that takes a byte slice representing flash memory
// For actual TinyGo implementations, you would:
// 1. Use platform-specific flash APIs
// 2. Map flash regions using linker scripts
// 3. Handle flash write operations (which are slow and have limited cycles)
//
// Example usage for platforms with machine.FlashData:
//
//	flashData := machine.FlashData()
//	factory := NewFlashMapFromData(flashData, true) // read-only
//
// Note: Some microcontrollers allow writing to flash, but:
// - Writes are slow (milliseconds per page)
// - Limited write cycles (typically 10,000-100,000)
// - Requires erase before write (page/sector level)
// - May require special power conditions
func NewFlashMap() types.MappedStorageFactory {
	// This is a placeholder - actual implementation needs platform-specific code
	// For now, return an error indicating it needs platform-specific setup
	return func(path string, readOnly bool) (types.MappedStorage, error) {
		// In a real implementation, you would:
		// 1. Parse path to determine flash region/sector
		// 2. Use platform-specific APIs to access flash
		// 3. Handle read-only vs writable flash regions
		// 4. Implement proper flash write/erase operations

		// For now, return an error with guidance
		return nil, fmt.Errorf("flash storage: platform-specific implementation required. " +
			"Use NewFlashMapFromData() with a byte slice, or implement platform-specific flash access")
	}
}

// NewFlashMapFromData creates a flash storage factory from a byte slice
// This is useful for testing or when flash is already accessible as a byte slice
// In real TinyGo implementations, you would get this from platform-specific APIs
func NewFlashMapFromData(data []byte, readOnly bool) types.MappedStorageFactory {
	return func(path string, ro bool) (types.MappedStorage, error) {
		// Use the readOnly parameter from the factory, not the function parameter
		// This allows the factory to specify read-only behavior
		return newFlashStorage(data, readOnly), nil
	}
}

// getFlashPointer is a helper for accessing flash memory via unsafe pointer
// This is used when flash is mapped at a specific memory address
// WARNING: This is unsafe and platform-specific
func getFlashPointer(addr uintptr, size int) []byte {
	// Convert pointer to byte slice
	// This assumes the flash memory is already mapped at the given address
	// In real implementations, you would use platform-specific methods
	ptr := (*[1 << 30]byte)(unsafe.Pointer(addr))
	return ptr[:size:size]
}
