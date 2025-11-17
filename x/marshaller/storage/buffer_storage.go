//go:build !tinygo

package storage

import (
	"archive/tar"
	"bytes"
	"io"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// bufferStorage wraps memoryStorage but keeps the underlying buffer alive even
// after Close is called, enabling reusable in-memory writers.
type bufferStorage struct {
	storage *memoryStorage
}

func newBufferStorage(initial []byte) *bufferStorage {
	buf := newMemoryStorage(int64(len(initial)))
	copy(buf.data, initial)
	return &bufferStorage{storage: buf}
}

func (b *bufferStorage) Map(offset, length int64) (types.MappedRegion, error) {
	return b.storage.Map(offset, length)
}

func (b *bufferStorage) Size() (int64, error) {
	return b.storage.Size()
}

func (b *bufferStorage) Grow(size int64) error {
	return b.storage.Grow(size)
}

func (b *bufferStorage) Close() error {
	// No-op: keep underlying buffer alive for later commits.
	return nil
}

func (b *bufferStorage) ReaderWriterSeeker() (io.ReadWriteSeeker, error) {
	return b.storage.ReaderWriterSeeker()
}

func (b *bufferStorage) writeToTar(tw *tar.Writer, entryName string) error {
	size, err := b.Size()
	if err != nil {
		return err
	}

	header := &tar.Header{
		Name: entryName,
		Mode: 0600,
		Size: size,
	}
	if err := tw.WriteHeader(header); err != nil {
		return err
	}

	region, err := b.Map(0, 0)
	if err != nil {
		return err
	}
	defer region.Unmap()

	reader := bytes.NewReader(region.Bytes())
	_, err = io.Copy(tw, reader)
	return err
}
