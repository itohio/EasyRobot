//go:build !tinygo

package storage

import (
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestFileStorage(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.dat")

	// Create file with initial data
	file, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Write initial data
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
	}
	if _, err := file.Write(data); err != nil {
		t.Fatalf("Failed to write initial data: %v", err)
	}
	file.Close()

	// Open storage
	file, err = os.OpenFile(filePath, os.O_RDWR, 0644)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}

	storage, err := newFileStorage(file, false)
	if err != nil {
		t.Fatalf("newFileStorage() failed: %v", err)
	}

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

	// Test reading from region
	regionData := region.Bytes()
	if len(regionData) != 512 {
		t.Errorf("Expected region data length 512, got %d", len(regionData))
	}

	// Test writing to region
	for i := range regionData {
		regionData[i] = byte((i + 100) % 256)
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

	// Verify file was written correctly
	file, err = os.Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file: %v", err)
	}
	defer file.Close()

	readData := make([]byte, 512)
	if _, err := file.Read(readData); err != nil {
		t.Fatalf("Failed to read file: %v", err)
	}

	// Check that our writes persisted
	for i := range readData {
		expected := byte((i + 100) % 256)
		if readData[i] != expected {
			t.Errorf("Expected byte %d at offset %d, got %d", expected, i, readData[i])
		}
	}
}

func TestFileStorageReadOnly(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_ro.dat")

	// Create file
	file, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	file.Close()

	// Open read-only
	file, err = os.Open(filePath)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}

	storage, err := newFileStorage(file, true)
	if err != nil {
		t.Fatalf("newFileStorage() failed: %v", err)
	}

	// Test Grow on read-only storage
	err = storage.Grow(2048)
	if err == nil {
		t.Error("Expected error when growing read-only storage")
	}

	storage.Close()
}

func TestFileStorageMapErrors(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_errors.dat")

	file, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	file.Close()

	file, err = os.OpenFile(filePath, os.O_RDWR, 0644)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}

	storage, err := newFileStorage(file, false)
	if err != nil {
		t.Fatalf("newFileStorage() failed: %v", err)
	}
	defer storage.Close()

	// Test invalid offset
	_, err = storage.Map(-1, 100)
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
}

func TestNewFileMap(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_factory.dat")

	factory := NewFileMap()

	storage, err := factory(filePath, false)
	if err != nil {
		t.Fatalf("NewFileMap() failed: %v", err)
	}
	defer storage.Close()

	size, err := storage.Size()
	if err != nil {
		t.Fatalf("Size() failed: %v", err)
	}
	if size != 0 {
		t.Errorf("Expected initial size 0 for new file, got %d", size)
	}
}

func TestFileStorageReaderWriterSeeker(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_rws.dat")

	file, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	file.Close()

	file, err = os.OpenFile(filePath, os.O_RDWR, 0644)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}

	storage, err := newFileStorage(file, false)
	if err != nil {
		t.Fatalf("newFileStorage() failed: %v", err)
	}
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
}

