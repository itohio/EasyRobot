package mnist

import (
	"compress/gzip"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Sample represents a single MNIST sample.
type Sample struct {
	Label int
	Image tensor.Tensor // Shape: [1, 28, 28] for single channel, 28x28 image
}

// Load loads MNIST dataset from CSV file.
// Format: label, pix-11, pix-12, ..., pix-28-28 (784 pixels total)
// Returns samples and any error.
func Load(filename string, maxSamples int) ([]Sample, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filename, err)
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzipReader.Close()

	reader := csv.NewReader(gzipReader)
	var samples []Sample

	rowNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read CSV row %d: %w", rowNum, err)
		}

		if len(record) < 785 {
			return nil, fmt.Errorf("row %d: expected at least 785 columns (label + 784 pixels), got %d", rowNum, len(record))
		}

		// Parse label (first column)
		label, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, fmt.Errorf("row %d: failed to parse label: %w", rowNum, err)
		}

		// Filter only digits 0-9 (focus on speed as requested)
		if label < 0 || label > 9 {
			rowNum++
			continue
		}

		// Parse pixels (remaining 784 columns)
		pixels := make([]float32, 784)
		for i := 0; i < 784; i++ {
			pixel, err := strconv.Atoi(record[i+1])
			if err != nil {
				return nil, fmt.Errorf("row %d, pixel %d: failed to parse pixel value: %w", rowNum, i, err)
			}
			// Normalize to [0, 1] range
			pixels[i] = float32(pixel) / 255.0
		}

		// Reshape to [1, 28, 28] (single channel, 28x28 image)
		// For Conv2D, we need [1, 1, 28, 28] format, but we'll add batch dimension later
		image := tensor.FromFloat32(tensor.NewShape(1, 28, 28), pixels)

		samples = append(samples, Sample{
			Label: label,
			Image: image,
		})

		rowNum++

		// Limit samples if maxSamples is specified
		if maxSamples > 0 && len(samples) >= maxSamples {
			break
		}
	}

	return samples, nil
}
