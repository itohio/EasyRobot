package render

import (
	"context"
	"fmt"
	"image"
	"image/color"

	"github.com/itohio/EasyRobot/types/spectrometer"
	"github.com/itohio/EasyRobot/x/math"
	"github.com/itohio/EasyRobot/x/math/colorscience"
	"github.com/itohio/EasyRobot/x/math/vec"
	"gocv.io/x/gocv"
)

// Renderer renders spectrum visualizations
type Renderer interface {
	// RenderSpectrum renders a spectrum graph
	RenderSpectrum(ctx context.Context, img gocv.Mat, spd colorscience.SPD, config *spectrometer.DisplaySettings) error

	// RenderColorimetry renders colorimetry overlay (XYZ, LAB, RGB HEX)
	RenderColorimetry(ctx context.Context, img gocv.Mat, xyz colorscience.XYZ, lab colorscience.LAB, rgb colorscience.RGB, displayMode ColorimetryDisplayMode) error

	// RenderColorSwatch renders an RGB color swatch
	RenderColorSwatch(ctx context.Context, img gocv.Mat, rgb colorscience.RGB, x, y, width, height int) error

	// RenderPatchComparison renders a patch comparison (half reference, half measured)
	RenderPatchComparison(ctx context.Context, img gocv.Mat, refPatch, measuredPatch Patch, x, y, width, height int) error
}

// renderer implements the Renderer interface
type renderer struct {
	// Default display settings if config is nil
	defaultWidth  int
	defaultHeight int
}

// NewRenderer creates a new renderer instance
func NewRenderer() Renderer {
	return &renderer{
		defaultWidth:  800,
		defaultHeight: 600,
	}
}

// RenderSpectrum renders a spectrum graph (wavelength vs. intensity plot)
func (r *renderer) RenderSpectrum(ctx context.Context, img gocv.Mat, spd colorscience.SPD, config *spectrometer.DisplaySettings) error {
	if img.Empty() {
		return fmt.Errorf("image is empty")
	}

	wavelengths := spd.Wavelengths()
	values := spd.Values()
	if wavelengths == nil || values == nil || wavelengths.Len() == 0 {
		return fmt.Errorf("SPD is empty")
	}

	imgSize := img.Size()
	imgWidth := imgSize[1]
	imgHeight := imgSize[0]

	// Calculate plot area (with margins)
	marginLeft := 60
	marginRight := 20
	marginTop := 20
	marginBottom := 40
	plotX := marginLeft
	plotY := marginTop
	plotWidth := imgWidth - marginLeft - marginRight
	plotHeight := imgHeight - marginTop - marginBottom

	if plotWidth <= 0 || plotHeight <= 0 {
		return fmt.Errorf("plot area too small")
	}

	// Find min/max wavelengths and values for scaling
	wlMin := wavelengths[0]
	wlMax := wavelengths[0]
	valMin := values[0]
	valMax := values[0]

	for i := 0; i < wavelengths.Len(); i++ {
		if wavelengths[i] < wlMin {
			wlMin = wavelengths[i]
		}
		if wavelengths[i] > wlMax {
			wlMax = wavelengths[i]
		}
		if values[i] < valMin {
			valMin = values[i]
		}
		if values[i] > valMax {
			valMax = values[i]
		}
	}

	// Handle edge case where all values are the same
	if valMax == valMin {
		valMax = valMin + 1.0
	}

	// Draw grid
	gridColor := color.RGBA{100, 100, 100, 255}
	numGridLines := 5
	for i := 0; i <= numGridLines; i++ {
		// Horizontal grid lines (wavelength)
		y := plotY + (plotHeight*i)/numGridLines
		gocv.Line(&img, image.Pt(plotX, y), image.Pt(plotX+plotWidth, y), gridColor, 1)

		// Vertical grid lines (intensity)
		x := plotX + (plotWidth*i)/numGridLines
		gocv.Line(&img, image.Pt(x, plotY), image.Pt(x, plotY+plotHeight), gridColor, 1)
	}

	// Draw axis labels
	labelColor := color.RGBA{255, 255, 255, 255}
	fontScale := 0.5
	thickness := 1

	// X-axis labels (wavelength)
	for i := 0; i <= numGridLines; i++ {
		wl := wlMin + (wlMax-wlMin)*float32(i)/float32(numGridLines)
		x := plotX + (plotWidth*i)/numGridLines
		label := fmt.Sprintf("%.0f", wl)
		gocv.PutText(&img, label, image.Pt(x-10, plotY+plotHeight+20), gocv.FontHersheyPlain, fontScale, labelColor, thickness)
	}

	// Y-axis labels (intensity)
	for i := 0; i <= numGridLines; i++ {
		val := valMax - (valMax-valMin)*float32(i)/float32(numGridLines)
		y := plotY + (plotHeight*i)/numGridLines
		label := fmt.Sprintf("%.2f", val)
		gocv.PutText(&img, label, image.Pt(5, y+5), gocv.FontHersheyPlain, fontScale, labelColor, thickness)
	}

	// Draw spectrum line with color-coded segments
	if wavelengths.Len() > 1 {
		for i := 0; i < wavelengths.Len()-1; i++ {
			wl0 := wavelengths[i]
			wl1 := wavelengths[i+1]
			val0 := values[i]
			val1 := values[i+1]

			// Convert wavelength to pixel coordinates
			x0 := plotX + int((wl0-wlMin)/(wlMax-wlMin)*float32(plotWidth))
			x1 := plotX + int((wl1-wlMin)/(wlMax-wlMin)*float32(plotWidth))
			y0 := plotY + plotHeight - int((val0-valMin)/(valMax-valMin)*float32(plotHeight))
			y1 := plotY + plotHeight - int((val1-valMin)/(valMax-valMin)*float32(plotHeight))

			// Get color for wavelength (average of two points)
			avgWl := (wl0 + wl1) / 2.0
			rgb := wavelengthToRGB(avgWl)
			lineColor := color.RGBA{
				R: uint8(rgb[0]),
				G: uint8(rgb[1]),
				B: uint8(rgb[2]),
				A: 255,
			}

			gocv.Line(&img, image.Pt(x0, y0), image.Pt(x1, y1), lineColor, 2)
		}
	}

	return nil
}

// RenderColorimetry renders colorimetry overlay (XYZ, LAB, RGB HEX)
func (r *renderer) RenderColorimetry(ctx context.Context, img gocv.Mat, xyz colorscience.XYZ, lab colorscience.LAB, rgb colorscience.RGB, displayMode ColorimetryDisplayMode) error {
	if img.Empty() {
		return fmt.Errorf("image is empty")
	}

	// Format colorimetry text based on display mode
	text := FormatColorimetry(xyz, lab, rgb, displayMode)

	// Draw semi-transparent background
	imgSize := img.Size()
	imgWidth := imgSize[1]

	// Position overlay in top-right corner
	padding := 10
	textX := imgWidth - 200
	textY := 30
	bgX := textX - padding
	bgY := textY - 20
	bgWidth := 190
	bgHeight := 30

	// Draw semi-transparent rectangle (approximate with darker rectangle)
	bgColor := color.RGBA{0, 0, 0, 200}
	gocv.Rectangle(&img, image.Rect(bgX, bgY, bgX+bgWidth, bgY+bgHeight), bgColor, -1)

	// Draw text
	textColor := color.RGBA{255, 255, 255, 255}
	fontScale := 0.6
	thickness := 1
	gocv.PutText(&img, text, image.Pt(textX, textY), gocv.FontHersheyPlain, fontScale, textColor, thickness)

	return nil
}

// RenderColorSwatch renders an RGB color swatch
func (r *renderer) RenderColorSwatch(ctx context.Context, img gocv.Mat, rgb colorscience.RGB, x, y, width, height int) error {
	if img.Empty() {
		return fmt.Errorf("image is empty")
	}

	R, G, B := vec.Vector3D(rgb).XYZ()
	// Clamp to 0-255 range
	R = math.Clamp(R, 0, 255)
	G = math.Clamp(G, 0, 255)
	B = math.Clamp(B, 0, 255)

	swatchColor := color.RGBA{
		R: uint8(R),
		G: uint8(G),
		B: uint8(B),
		A: 255,
	}

	// Draw filled rectangle
	gocv.Rectangle(&img, image.Rect(x, y, x+width, y+height), swatchColor, -1)

	// Draw border
	borderColor := color.RGBA{255, 255, 255, 255}
	gocv.Rectangle(&img, image.Rect(x, y, x+width, y+height), borderColor, 2)

	return nil
}

// RenderPatchComparison renders a patch comparison (half reference, half measured)
func (r *renderer) RenderPatchComparison(ctx context.Context, img gocv.Mat, refPatch, measuredPatch Patch, x, y, width, height int) error {
	if img.Empty() {
		return fmt.Errorf("image is empty")
	}

	// Calculate delta E if not already calculated
	if measuredPatch.DeltaE == 0 {
		measuredPatch.CalculateDeltaE()
	}

	// Draw left half (reference) - compute RGB from ReferenceXYZ
	refRGB := refPatch.ReferenceXYZ.ToRGB(true)
	R, G, B := vec.Vector3D(refRGB).XYZ()
	R = math.Clamp(R, 0, 255)
	G = math.Clamp(G, 0, 255)
	B = math.Clamp(B, 0, 255)
	refColor := color.RGBA{
		R: uint8(R),
		G: uint8(G),
		B: uint8(B),
		A: 255,
	}
	gocv.Rectangle(&img, image.Rect(x, y, x+width/2, y+height), refColor, -1)

	// Draw right half (measured) - compute RGB from MeasuredXYZ
	measRGB := measuredPatch.MeasuredXYZ.ToRGB(true)
	R, G, B = vec.Vector3D(measRGB).XYZ()
	R = math.Clamp(R, 0, 255)
	G = math.Clamp(G, 0, 255)
	B = math.Clamp(B, 0, 255)
	measColor := color.RGBA{
		R: uint8(R),
		G: uint8(G),
		B: uint8(B),
		A: 255,
	}
	gocv.Rectangle(&img, image.Rect(x+width/2, y, x+width, y+height), measColor, -1)

	// Draw border
	borderColor := color.RGBA{255, 255, 255, 255}
	gocv.Rectangle(&img, image.Rect(x, y, x+width, y+height), borderColor, 2)

	// Draw dividing line
	dividerColor := color.RGBA{0, 0, 0, 255}
	gocv.Line(&img, image.Pt(x+width/2, y), image.Pt(x+width/2, y+height), dividerColor, 2)

	// Draw patch name and delta E
	textColor := color.RGBA{255, 255, 255, 255}
	fontScale := 0.5
	thickness := 1

	// Patch name
	if measuredPatch.Name != "" {
		gocv.PutText(&img, measuredPatch.Name, image.Pt(x+5, y+15), gocv.FontHersheyPlain, fontScale, textColor, thickness)
	}

	// Delta E
	deltaEText := fmt.Sprintf("ΔE: %.2f", measuredPatch.DeltaE)
	gocv.PutText(&img, deltaEText, image.Pt(x+5, y+height-5), gocv.FontHersheyPlain, fontScale, textColor, thickness)

	return nil
}

// wavelengthToRGB converts a wavelength (in nanometers) to approximate RGB color for visualization
// This is a simplified approximation for spectrum visualization
func wavelengthToRGB(wavelength float32) vec.Vector3D {
	// Clamp wavelength to visible range (380-780 nm)
	wl := math.Clamp(wavelength, 380, 780)

	var R, G, B float32

	if wl < 440 {
		R = -(wl - 440) / (440 - 380)
		G = 0.0
		B = 1.0
	} else if wl < 490 {
		R = 0.0
		G = (wl - 440) / (490 - 440)
		B = 1.0
	} else if wl < 510 {
		R = 0.0
		G = 1.0
		B = -(wl - 510) / (510 - 490)
	} else if wl < 580 {
		R = (wl - 510) / (580 - 510)
		G = 1.0
		B = 0.0
	} else if wl < 645 {
		R = 1.0
		G = -(wl - 645) / (645 - 580)
		B = 0.0
	} else {
		R = 1.0
		G = 0.0
		B = 0.0
	}

	// Adjust brightness for edges of visible spectrum
	var factor float32
	if wl > 700 {
		factor = 0.3 + 0.7*(780-wl)/(780-700)
	} else if wl < 420 {
		factor = 0.3 + 0.7*(wl-380)/(420-380)
	} else {
		factor = 1.0
	}

	R *= factor
	G *= factor
	B *= factor

	// Convert to 0-255 range
	R = math.Clamp(R*255, 0, 255)
	G = math.Clamp(G*255, 0, 255)
	B = math.Clamp(B*255, 0, 255)

	return vec.Vector3D{R, G, B}
}
