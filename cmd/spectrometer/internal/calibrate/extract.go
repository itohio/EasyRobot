package calibrate

import (
	"fmt"
	"image"
	"math"

	"github.com/itohio/EasyRobot/x/math/colorscience"
	"github.com/itohio/EasyRobot/x/math/vec"
	cv "gocv.io/x/gocv"
)

// extractSpectrumFromBoundingBox extracts spectrum from the bounding box region
// Returns SPD with pixel coordinates as wavelengths (will need wavelength calibration later)
func extractSpectrumFromBoundingBox(img *cv.Mat, bbox *image.Rectangle, angle float64, flipped bool) (colorscience.SPD, error) {
	if img == nil || img.Empty() {
		return colorscience.SPD{}, fmt.Errorf("image is empty")
	}
	if bbox == nil {
		return colorscience.SPD{}, fmt.Errorf("bounding box is nil")
	}

	// Extract region from bounding box
	region := img.Region(*bbox)
	defer region.Close()

	// Handle rotation if needed
	var processedRegion cv.Mat
	if math.Abs(angle) > 1e-6 {
		// Rotate region to horizontal
		center := image.Pt(region.Cols()/2, region.Rows()/2)
		rotationMatrix := cv.GetRotationMatrix2D(center, -angle, 1.0)
		defer rotationMatrix.Close()

		// Calculate new size after rotation
		cosA := math.Cos(angle * math.Pi / 180.0)
		sinA := math.Sin(angle * math.Pi / 180.0)
		newWidth := int(math.Abs(float64(region.Cols())*cosA) + math.Abs(float64(region.Rows())*sinA))
		newHeight := int(math.Abs(float64(region.Cols())*sinA) + math.Abs(float64(region.Rows())*cosA))

		processedRegion = cv.NewMat()
		cv.WarpAffine(region, &processedRegion, rotationMatrix, image.Pt(newWidth, newHeight))
	} else {
		// No rotation needed, clone the region
		processedRegion = region.Clone()
	}
	defer processedRegion.Close()

	// Handle horizontal flip if needed
	if flipped {
		cv.Flip(processedRegion, &processedRegion, 1) // 1 = horizontal flip
	}

	// Convert to grayscale if needed
	var grayMat cv.Mat
	if processedRegion.Channels() > 1 {
		grayMat = cv.NewMat()
		cv.CvtColor(processedRegion, &grayMat, cv.ColorBGRToGray)
		defer grayMat.Close()
	} else {
		grayMat = processedRegion
	}

	// Average rows to get intensity per x coordinate
	width := grayMat.Cols()
	height := grayMat.Rows()

	if width == 0 || height == 0 {
		return colorscience.SPD{}, fmt.Errorf("processed region has zero size")
	}

	// Create intensity vector (one value per x coordinate)
	intensity := vec.New(width)

	// Average all rows for each column
	for x := 0; x < width; x++ {
		var sum float32
		for y := 0; y < height; y++ {
			val := grayMat.GetUCharAt(y, x)
			sum += float32(val)
		}
		intensity[x] = sum / float32(height)
	}

	// Create wavelength vector (pixel coordinates, will be calibrated later)
	// For now, use pixel index as wavelength placeholder
	wavelengths := vec.New(width)
	for i := 0; i < width; i++ {
		wavelengths[i] = float32(i) // Pixel coordinate, will be calibrated to actual wavelength
	}

	// Create SPD from wavelengths and intensities
	spd := colorscience.NewSPD(wavelengths, intensity)

	return spd, nil
}

