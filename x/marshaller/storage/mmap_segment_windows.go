//go:build !tinygo && windows

package storage

import (
	"errors"
	"os"
	"sync"
	"syscall"
	"unsafe"
)

var (
	allocationOnce   sync.Once
	allocGranularity int64 = 65536 // sane default
)

func systemAllocationGranularity() int64 {
	allocationOnce.Do(func() {
		var info syscall.SystemInfo
		syscall.GetSystemInfo(&info)
		allocGranularity = int64(info.AllocationGranularity)
	})
	return allocGranularity
}

func mapFileSegment(file *os.File, offset int64, length int, readOnly bool) (*mappedSegment, error) {
	if length <= 0 {
		return nil, errors.New("mmap: length must be positive")
	}

	gran := systemAllocationGranularity()
	pageOffset := offset / gran * gran
	delta := int(offset - pageOffset)
	mapLength := length + delta
	if mapLength <= 0 {
		return nil, errors.New("mmap: map length must be positive")
	}

	var protect uint32 = syscall.PAGE_READONLY
	var access uint32 = syscall.FILE_MAP_READ
	if !readOnly {
		protect = syscall.PAGE_READWRITE
		access = syscall.FILE_MAP_WRITE
	}

	maxSize := pageOffset + int64(mapLength)
	handle, err := syscall.CreateFileMapping(syscall.Handle(file.Fd()), nil, protect,
		uint32(uint64(maxSize)>>32), uint32(uint64(maxSize)&0xffffffff), nil)
	if err != nil {
		return nil, err
	}

	addr, err := syscall.MapViewOfFile(handle, access,
		uint32(uint64(pageOffset)>>32), uint32(uint64(pageOffset)&0xffffffff), uintptr(mapLength))
	if err != nil {
		syscall.CloseHandle(handle)
		return nil, err
	}

	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), mapLength)

	return &mappedSegment{
		base:   data,
		view:   data[delta : delta+length],
		handle: uintptr(handle),
		addr:   addr,
		length: mapLength,
	}, nil
}

func syncSegment(seg *mappedSegment) error {
	if seg == nil || len(seg.base) == 0 {
		return nil
	}
	if err := syscall.FlushViewOfFile(seg.addr, uintptr(seg.length)); err != nil {
		return err
	}
	return nil
}

func unmapSegment(seg *mappedSegment) error {
	if seg == nil || len(seg.base) == 0 {
		return nil
	}
	if err := syscall.UnmapViewOfFile(seg.addr); err != nil {
		return err
	}
	if seg.handle != 0 {
		return syscall.CloseHandle(syscall.Handle(seg.handle))
	}
	return nil
}
