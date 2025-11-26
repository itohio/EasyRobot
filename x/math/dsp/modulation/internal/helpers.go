package internal

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
)

// processInPlace processes data using one of two strategies:
// 1. If dst implements Viewer interface, obtain the slice and process in-place
// 2. Otherwise, read from src in small batches and write to dst
func processInPlace(
	dst buffer.Writer[complex64],
	src buffer.Reader[complex64],
	batchSize int,
	processFn func([]complex64),
) int {
	if dst == nil {
		return 0
	}

	// Strategy 1: If dst implements Viewer, process in-place
	if viewer, ok := dst.(buffer.Viewer[complex64]); ok {
		items := viewer.Items()
		if len(items) == 0 {
			return 0
		}
		processFn(items)
		return len(items)
	}

	// Strategy 2: Read from src in batches and write to dst
	if src == nil {
		return 0
	}

	total := 0
	batch := make([]complex64, batchSize)

	for {
		// Read batch from src
		n, err := src.Read(batch)
		if n == 0 {
			if err != nil && err != io.EOF {
				break
			}
			break
		}

		// Process batch
		processFn(batch[:n])

		// Write processed batch to dst
		written, writeErr := dst.Write(batch[:n])
		total += written

		if writeErr != nil {
			break
		}
		if n < len(batch) || err == io.EOF {
			break
		}
	}

	return total
}
