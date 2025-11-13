package gocv

import (
	"encoding/binary"
	"fmt"
	"os"

	cv "gocv.io/x/gocv"
)

const (
	matMagic uint64 = 0xabcdef0012345678
)

func readCustomMat(path string) (cv.Mat, error) {
	fp, err := os.Open(path)
	if err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: open mat file: %w", err)
	}
	defer fp.Close()

	var (
		magic    uint64
		lenSizes uint8
		matType  cv.MatType
		lenBytes uint64
	)

	if err := binary.Read(fp, binary.LittleEndian, &magic); err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: read mat header: %w", err)
	}
	if magic != matMagic {
		return cv.Mat{}, fmt.Errorf("gocv: invalid mat magic")
	}

	if err := binary.Read(fp, binary.LittleEndian, &lenSizes); err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: read mat dimension count: %w", err)
	}

	sizes := make([]int32, lenSizes)
	if err := binary.Read(fp, binary.LittleEndian, &sizes); err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: read mat dimensions: %w", err)
	}

	if err := binary.Read(fp, binary.LittleEndian, &matType); err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: read mat type: %w", err)
	}

	if err := binary.Read(fp, binary.LittleEndian, &lenBytes); err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: read mat data length: %w", err)
	}

	bytes := make([]byte, lenBytes)
	if err := binary.Read(fp, binary.LittleEndian, bytes); err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: read mat payload: %w", err)
	}

	dims := make([]int, len(sizes))
	for i, s := range sizes {
		dims[i] = int(s)
	}

	mat, err := cv.NewMatWithSizesFromBytes(dims, matType, bytes)
	if err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: constructing mat: %w", err)
	}
	return mat, nil
}
