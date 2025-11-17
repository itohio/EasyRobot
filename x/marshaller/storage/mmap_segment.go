//go:build !tinygo

package storage

type mappedSegment struct {
	base   []byte
	view   []byte
	handle uintptr
	addr   uintptr
	length int
}
