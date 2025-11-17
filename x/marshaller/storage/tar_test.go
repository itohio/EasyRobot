//go:build !tinygo

package storage

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func TestTarStoragePlainTar(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "graph.tar")
	payload := []byte("the quick brown fox jumps over the lazy dog")

	writeTarArchive(t, archivePath, false, map[string][]byte{
		"nodes.graph": payload,
	})

	tarFactory := NewTarMap(archivePath)
	factory := tarFactory.Factory()
	logicalPath := filepath.Join(tmpDir, "nodes.graph")
	storage, err := factory(logicalPath, true)
	if err != nil {
		t.Fatalf("NewTarMap() failed: %v", err)
	}
	defer storage.Close()

	size, err := storage.Size()
	if err != nil {
		t.Fatalf("Size() failed: %v", err)
	}
	if size != int64(len(payload)) {
		t.Fatalf("unexpected size %d", size)
	}

	region, err := storage.Map(0, 0)
	if err != nil {
		t.Fatalf("Map() failed: %v", err)
	}
	defer region.Unmap()

	if got := string(region.Bytes()); got != string(payload) {
		t.Fatalf("unexpected payload: %s", got)
	}

	rwsStorage, ok := storage.(types.ReaderWriterSeekerStorage)
	if !ok {
		t.Fatalf("tar storage should implement ReaderWriterSeekerStorage")
	}

	rws, err := rwsStorage.ReaderWriterSeeker()
	if err != nil {
		t.Fatalf("ReaderWriterSeeker() failed: %v", err)
	}

	buf := make([]byte, len(payload))
	if _, err := rws.Read(buf); err != nil && err != io.EOF {
		t.Fatalf("read failed: %v", err)
	}
	if string(buf) != string(payload) {
		t.Fatalf("unexpected reader payload")
	}

	if _, err := rws.Write([]byte("fail")); err == nil {
		t.Fatalf("expected write to fail on read-only storage")
	}
}

func TestTarStorageGzip(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "graph.tar.gz")
	payload := []byte("gzip nodes data")

	writeTarArchive(t, archivePath, true, map[string][]byte{
		"nodes.graph": payload,
	})

	tarFactory := NewTarMap(archivePath)
	factory := tarFactory.Factory()
	logicalPath := filepath.Join(tmpDir, "nodes.graph")
	storage, err := factory(logicalPath, true)
	if err != nil {
		t.Fatalf("NewTarMap() failed: %v", err)
	}
	defer storage.Close()

	region, err := storage.Map(2, int64(len(payload))-2)
	if err != nil {
		t.Fatalf("Map() failed: %v", err)
	}
	defer region.Unmap()

	if string(region.Bytes()) != string(payload[2:]) {
		t.Fatalf("unexpected mapped bytes")
	}
}

func TestTarStorageMissingEntry(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "graph.tar")

	writeTarArchive(t, archivePath, false, map[string][]byte{
		"nodes.graph": []byte("payload"),
	})

	tarFactory := NewTarMap(archivePath)
	factory := tarFactory.Factory()
	logicalPath := filepath.Join(tmpDir, "edges.graph")
	_, err := factory(logicalPath, true)
	if err == nil {
		t.Fatalf("expected error for missing entry")
	}
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("expected fs.ErrNotExist, got %v", err)
	}
}

func TestTarStorageInvalidPath(t *testing.T) {
	tarFactory := NewTarMap("")
	factory := tarFactory.Factory()
	_, err := factory("/tmp/does-not-exist.graph", true)
	if err == nil {
		t.Fatalf("expected error when archive path missing and file not present")
	}
}

func TestTarStorageFallbackToFile(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "nodes.graph")
	payload := []byte("disk file data")
	if err := os.WriteFile(filePath, payload, 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	tarFactory := NewTarMap(filepath.Join(tmpDir, "unused.tar"))
	factory := tarFactory.Factory()
	storage, err := factory(filePath, false)
	if err != nil {
		t.Fatalf("expected direct file storage, got error: %v", err)
	}
	defer storage.Close()

	region, err := storage.Map(0, 0)
	if err != nil {
		t.Fatalf("Map() failed: %v", err)
	}
	defer region.Unmap()

	if got := string(region.Bytes()); got != string(payload) {
		t.Fatalf("unexpected contents: %s", got)
	}
}

func TestTarFactoryCommit(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "graph.tar")
	original := []byte("aaaaaaaaaaaa")
	writeTarArchive(t, archivePath, false, map[string][]byte{
		"nodes.graph": original,
	})

	tarFactory := NewTarMap(archivePath)
	factory := tarFactory.Factory()
	logicalPath := filepath.Join(tmpDir, "nodes.graph")

	storage, err := factory(logicalPath, false)
	if err != nil {
		t.Fatalf("factory writable storage failed: %v", err)
	}

	region, err := storage.Map(0, 0)
	if err != nil {
		t.Fatalf("Map() failed: %v", err)
	}
	copy(region.Bytes(), []byte("bbbbbbbbbbbb"))
	region.Unmap()
	if err := storage.Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}

	if err := tarFactory.Commit(); err != nil {
		t.Fatalf("Commit() failed: %v", err)
	}

	roStorage, err := factory(logicalPath, true)
	if err != nil {
		t.Fatalf("factory read-only storage failed: %v", err)
	}
	defer roStorage.Close()

	readRegion, err := roStorage.Map(0, 0)
	if err != nil {
		t.Fatalf("Map() failed: %v", err)
	}
	defer readRegion.Unmap()

	if got := string(readRegion.Bytes()); got != "bbbbbbbbbbbb" {
		t.Fatalf("expected updated data, got %s", got)
	}
}

func writeTarArchive(t *testing.T, path string, gzipEnabled bool, entries map[string][]byte) {
	t.Helper()

	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create archive: %v", err)
	}
	defer file.Close()

	var writer io.Writer = file
	var gzipWriter *gzip.Writer
	if gzipEnabled {
		gzipWriter = gzip.NewWriter(file)
		writer = gzipWriter
	}

	tw := tar.NewWriter(writer)
	for name, data := range entries {
		if err := tw.WriteHeader(&tar.Header{
			Name: name,
			Mode: 0600,
			Size: int64(len(data)),
		}); err != nil {
			t.Fatalf("write header: %v", err)
		}
		if _, err := tw.Write(data); err != nil {
			t.Fatalf("write entry: %v", err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar writer: %v", err)
	}
	if gzipWriter != nil {
		if err := gzipWriter.Close(); err != nil {
			t.Fatalf("close gzip writer: %v", err)
		}
	}
}
