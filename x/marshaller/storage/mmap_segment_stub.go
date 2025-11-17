//go:build !tinygo && !unix

package storage

import (
	"errors"
	"os"
)

var errUnsupportedMMap = errors.New("mmap: not supported on this platform")

func mapFileSegment(file *os.File, offset int64, length int, readOnly bool) (*mappedSegment, error) {
	return nil, errUnsupportedMMap
}

func syncSegment(seg *mappedSegment) error {
	return errUnsupportedMMap
}

func unmapSegment(seg *mappedSegment) error {
	return errUnsupportedMMap
}
