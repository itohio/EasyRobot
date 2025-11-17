package storage

import (
	"io"
	"testing"
)

func TestMemoryStorage(t *testing.T) {
	storage := newMemoryStorage(1024)

	// Test Size
	size, err := storage.Size()
	if err != nil {
		t.Fatalf("Size() failed: %v", err)
	}
	if size != 1024 {
		t.Errorf("Expected size 1024, got %d", size)
	}

	// Test Map
	region, err := storage.Map(0, 512)
	if err != nil {
		t.Fatalf("Map() failed: %v", err)
	}
	if region.Size() != 512 {
		t.Errorf("Expected region size 512, got %d", region.Size())
	}

	// Test writing to region
	data := region.Bytes()
	for i := range data {
		data[i] = byte(i % 256)
	}

	// Test Sync
	if err := region.Sync(); err != nil {
		t.Errorf("Sync() failed: %v", err)
	}

	// Test Unmap
	if err := region.Unmap(); err != nil {
		t.Errorf("Unmap() failed: %v", err)
	}

	// Test Grow
	if err := storage.Grow(2048); err != nil {
		t.Fatalf("Grow() failed: %v", err)
	}

	size, err = storage.Size()
	if err != nil {
		t.Fatalf("Size() after Grow() failed: %v", err)
	}
	if size != 2048 {
		t.Errorf("Expected size 2048 after Grow(), got %d", size)
	}

	// Test Close
	if err := storage.Close(); err != nil {
		t.Errorf("Close() failed: %v", err)
	}

	// Test operations after close
	_, err = storage.Size()
	if err == nil {
		t.Error("Expected error after Close(), got nil")
	}
}

func TestMemoryStorageMapErrors(t *testing.T) {
	storage := newMemoryStorage(1024)

	// Test invalid offset
	_, err := storage.Map(-1, 100)
	if err == nil {
		t.Error("Expected error for negative offset")
	}

	_, err = storage.Map(2000, 100)
	if err == nil {
		t.Error("Expected error for offset beyond size")
	}

	// Test invalid length
	_, err = storage.Map(0, 2000)
	if err == nil {
		t.Error("Expected error for length extending beyond size")
	}

	storage.Close()
}

func TestMemoryStorageGrowErrors(t *testing.T) {
	storage := newMemoryStorage(1024)

	// Test negative size
	err := storage.Grow(-1)
	if err == nil {
		t.Error("Expected error for negative size")
	}

	// Test grow to smaller size (should succeed, no-op)
	err = storage.Grow(512)
	if err != nil {
		t.Errorf("Grow to smaller size should succeed: %v", err)
	}

	storage.Close()
}

func TestNewMemoryMap(t *testing.T) {
	factory := NewMemoryMap()

	storage, err := factory("test", false)
	if err != nil {
		t.Fatalf("NewMemoryMap() failed: %v", err)
	}
	defer storage.Close()

	size, err := storage.Size()
	if err != nil {
		t.Fatalf("Size() failed: %v", err)
	}
	if size != 0 {
		t.Errorf("Expected initial size 0, got %d", size)
	}
}

func TestMemoryStorageReaderWriterSeeker(t *testing.T) {
	storage := newMemoryStorage(1024)
	defer storage.Close()

	// Test ReaderWriterSeeker
	rws, err := storage.ReaderWriterSeeker()
	if err != nil {
		t.Fatalf("ReaderWriterSeeker() failed: %v", err)
	}

	// Test Write
	testData := []byte("Hello, World!")
	n, err := rws.Write(testData)
	if err != nil {
		t.Fatalf("Write() failed: %v", err)
	}
	if n != len(testData) {
		t.Errorf("Expected to write %d bytes, wrote %d", len(testData), n)
	}

	// Test Seek to beginning
	pos, err := rws.Seek(0, io.SeekStart)
	if err != nil {
		t.Fatalf("Seek() failed: %v", err)
	}
	if pos != 0 {
		t.Errorf("Expected position 0, got %d", pos)
	}

	// Test Read
	readData := make([]byte, len(testData))
	n, err = rws.Read(readData)
	if err != nil && err != io.EOF {
		t.Fatalf("Read() failed: %v", err)
	}
	if n != len(testData) {
		t.Errorf("Expected to read %d bytes, read %d", len(testData), n)
	}
	if string(readData) != string(testData) {
		t.Errorf("Expected %s, got %s", string(testData), string(readData))
	}

	// Test Seek from current
	pos, err = rws.Seek(5, io.SeekCurrent)
	if err != nil {
		t.Fatalf("Seek() failed: %v", err)
	}
	if pos != int64(len(testData)+5) {
		t.Errorf("Expected position %d, got %d", len(testData)+5, pos)
	}

	// Test Seek from end
	pos, err = rws.Seek(-3, io.SeekEnd)
	if err != nil {
		t.Fatalf("Seek() failed: %v", err)
	}
	size, _ := storage.Size()
	expectedPos := size - 3
	if pos != expectedPos {
		t.Errorf("Expected position %d, got %d", expectedPos, pos)
	}
}

