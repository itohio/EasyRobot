//go:build !tinygo && unix

package storage

import (
	"errors"
	"os"
	"syscall"
	"unsafe"
)

func mapFileSegment(file *os.File, offset int64, length int, readOnly bool) (*mappedSegment, error) {
	if length <= 0 {
		return nil, errors.New("mmap: length must be positive")
	}

	pageSize := int64(os.Getpagesize())
	if pageSize == 0 {
		pageSize = 4096
	}

	pageOffset := offset / pageSize * pageSize
	delta := int(offset - pageOffset)
	mapLength := length + delta
	if mapLength <= 0 {
		return nil, errors.New("mmap: map length must be positive")
	}

	prot := syscall.PROT_READ
	if !readOnly {
		prot |= syscall.PROT_WRITE
	}

	data, err := syscall.Mmap(int(file.Fd()), pageOffset, mapLength, prot, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	return &mappedSegment{
		base: data,
		view: data[delta : delta+length],
	}, nil
}

func syncSegment(seg *mappedSegment) error {
	if seg == nil || len(seg.base) == 0 {
		return nil
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC,
		uintptr(unsafe.Pointer(&seg.base[0])),
		uintptr(len(seg.base)),
		uintptr(syscall.MS_SYNC))
	if errno != 0 {
		return errno
	}
	return nil
}

func unmapSegment(seg *mappedSegment) error {
	if seg == nil || len(seg.base) == 0 {
		return nil
	}
	return syscall.Munmap(seg.base)
}
