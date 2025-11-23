package calibrate

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"
	"sync"
	"sync/atomic"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/config"
	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/render"
	"github.com/itohio/EasyRobot/types/spectrometer"
	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/spf13/cobra"
	cv "gocv.io/x/gocv"
)

// BoundingBoxState manages the state of the bounding box during calibration
type BoundingBoxState struct {
	mu      sync.RWMutex
	bbox    *image.Rectangle // Fixed bounding box (nil if not fixed)
	angle   float64          // Rotation angle in degrees (for rotated strips)
	fixed   bool             // Whether bounding box is fixed
	flipped bool             // Whether bounding box is horizontally flipped
}

// Get returns a copy of the current bounding box state
func (s *BoundingBoxState) Get() (*image.Rectangle, bool, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.bbox == nil {
		return nil, s.fixed, s.flipped
	}
	bbox := *s.bbox
	return &bbox, s.fixed, s.flipped
}

// GetWithAngle returns a copy of the current bounding box state including rotation angle
func (s *BoundingBoxState) GetWithAngle() (*image.Rectangle, float64, bool, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.bbox == nil {
		return nil, 0.0, s.fixed, s.flipped
	}
	bbox := *s.bbox
	return &bbox, s.angle, s.fixed, s.flipped
}

// SetFixed sets the bounding box as fixed
func (s *BoundingBoxState) SetFixed(bbox *image.Rectangle) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if bbox != nil {
		b := *bbox
		s.bbox = &b
	}
	s.fixed = true
	slog.Info("Bounding box fixed", "bbox", s.bbox, "angle", s.angle)
}

// SetFixedWithAngle sets the bounding box as fixed with rotation angle
func (s *BoundingBoxState) SetFixedWithAngle(bbox *image.Rectangle, angle float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if bbox != nil {
		b := *bbox
		s.bbox = &b
	}
	s.angle = angle
	s.fixed = true
	slog.Info("Bounding box fixed", "bbox", s.bbox, "angle", s.angle)
}

// SetFixedWithAngleAndFlip sets the bounding box as fixed with rotation angle and flip state
func (s *BoundingBoxState) SetFixedWithAngleAndFlip(bbox *image.Rectangle, angle float64, flipped bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if bbox != nil {
		b := *bbox
		s.bbox = &b
	}
	s.angle = angle
	s.flipped = flipped
	s.fixed = true
	slog.Info("Bounding box fixed", "bbox", s.bbox, "angle", s.angle, "flipped", s.flipped)
}

// ToggleFlip toggles the horizontal flip state
func (s *BoundingBoxState) ToggleFlip() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.flipped = !s.flipped
	slog.Info("Bounding box flip toggled", "flipped", s.flipped)
}

// SetDetected updates the detected bounding box (only if not fixed)
func (s *BoundingBoxState) SetDetected(bbox *image.Rectangle, angle float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.fixed && bbox != nil {
		b := *bbox
		s.bbox = &b
		s.angle = angle
	}
}

// Reset unfixes the bounding box (allows new detection)
func (s *BoundingBoxState) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.fixed = false
	slog.Info("Bounding box reset (unfixed)")
}

// rectangleToSpectrumWindow converts an image.Rectangle to SpectrumWindow (center, size, rotation, flipped)
func rectangleToSpectrumWindow(rect *image.Rectangle, angle float64, flipped bool) *spectrometer.SpectrumWindow {
	if rect == nil {
		return nil
	}

	centerX := rect.Min.X + rect.Dx()/2
	centerY := rect.Min.Y + rect.Dy()/2

	return &spectrometer.SpectrumWindow{
		Center: &spectrometer.Point2D{
			X: int32(centerX),
			Y: int32(centerY),
		},
		Size: &spectrometer.Size2D{
			Width:  int32(rect.Dx()),
			Height: int32(rect.Dy()),
		},
		RotationDeg: angle, // Store actual rotation angle
		Flipped:     flipped, // Store flip state
	}
}

// spectrumWindowToRectangle converts a SpectrumWindow to image.Rectangle
// Properly handles rotated rectangles by computing axis-aligned bounding box
func spectrumWindowToRectangle(window *spectrometer.SpectrumWindow) *image.Rectangle {
	if window == nil || window.Center == nil || window.Size == nil {
		return nil
	}

	// Calculate top-left corner from center and size
	halfWidth := float64(window.Size.Width) / 2.0
	halfHeight := float64(window.Size.Height) / 2.0
	centerX := float64(window.Center.X)
	centerY := float64(window.Center.Y)
	angleRad := window.RotationDeg * math.Pi / 180.0

	// For axis-aligned rectangles (rotation = 0), use simple calculation
	if math.Abs(window.RotationDeg) < 1e-6 {
		return &image.Rectangle{
			Min: image.Pt(
				int(centerX-halfWidth),
				int(centerY-halfHeight),
			),
			Max: image.Pt(
				int(centerX+halfWidth),
				int(centerY+halfHeight),
			),
		}
	}

	// For rotated rectangles, compute the 4 corner points and find axis-aligned bounding box
	cosA := math.Cos(angleRad)
	sinA := math.Sin(angleRad)

	// Rotate the four corners relative to center
	corners := []struct {
		x, y float64
	}{
		{-halfWidth, -halfHeight}, // Top-left (relative to center)
		{halfWidth, -halfHeight},  // Top-right
		{halfWidth, halfHeight},   // Bottom-right
		{-halfWidth, halfHeight},  // Bottom-left
	}

	// Rotate each corner and translate to absolute coordinates
	rotatedCorners := make([]image.Point, 4)
	for i, corner := range corners {
		// Apply rotation
		rotX := corner.x*cosA - corner.y*sinA
		rotY := corner.x*sinA + corner.y*cosA
		// Translate to absolute coordinates
		rotatedCorners[i] = image.Pt(
			int(centerX+rotX),
			int(centerY+rotY),
		)
	}

	// Find axis-aligned bounding box from rotated corners
	minX := rotatedCorners[0].X
	maxX := rotatedCorners[0].X
	minY := rotatedCorners[0].Y
	maxY := rotatedCorners[0].Y

	for _, pt := range rotatedCorners[1:] {
		if pt.X < minX {
			minX = pt.X
		}
		if pt.X > maxX {
			maxX = pt.X
		}
		if pt.Y < minY {
			minY = pt.Y
		}
		if pt.Y > maxY {
			maxY = pt.Y
		}
	}

	return &image.Rectangle{
		Min: image.Pt(minX, minY),
		Max: image.Pt(maxX, maxY),
	}
}

// RunCameraCalibration runs the camera calibration command
// Continuously captures frames, detects bounding box, and displays with interaction support
func RunCameraCalibration(ctx context.Context, cmd *cobra.Command) error {
	slog.Info("Starting camera calibration")

	// Create cancelable context for graceful shutdown
	calibCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Get config flags
	configPath, _ := cmd.Flags().GetString("config")
	configOutputPath, _ := cmd.Flags().GetString("config-output")

	// Load config if provided
	var cfg *spectrometer.SpectrometerConfiguration
	if configPath != "" {
		loader := config.NewLoader("")
		loadedCfg, err := loader.Load(ctx, configPath)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}
		cfg = loadedCfg
		slog.Info("Config loaded", "path", configPath)
	} else {
		// Create new config if not provided
		cfg = &spectrometer.SpectrometerConfiguration{}
	}

	// Create camera source
	src, err := source.NewFromCobra(cmd)
	if err != nil {
		return fmt.Errorf("failed to create camera source: %w", err)
	}
	defer src.Close()

	// Start source
	if err := src.Start(calibCtx); err != nil {
		return fmt.Errorf("failed to start camera source: %w", err)
	}

	// Bounding box state (thread-safe)
	bboxState := &BoundingBoxState{}

	// Load bounding box from config if available
	if cfg.Window != nil {
		bbox := spectrumWindowToRectangle(cfg.Window)
		if bbox != nil {
			angle := cfg.Window.RotationDeg
			flipped := cfg.Window.Flipped
			bboxState.SetFixedWithAngleAndFlip(bbox, angle, flipped)
			slog.Info("Bounding box loaded from config",
				"x", bbox.Min.X,
				"y", bbox.Min.Y,
				"width", bbox.Dx(),
				"height", bbox.Dy(),
				"angle", angle,
				"flipped", flipped,
			)
		}
	}

	// Track if we need to save config when bounding box is fixed
	shouldSaveConfig := configOutputPath != ""
	var configSaved atomic.Bool

	// Get display settings from flags
	title, _ := cmd.Flags().GetString("title")
	if title == "" {
		title = "Camera Calibration"
	}
	width, _ := cmd.Flags().GetInt("window-width")
	height, _ := cmd.Flags().GetInt("window-height")

	// Mouse event handler - handle clicks to fix/flip bounding box
	onMouse := func(event types.MouseEvent) bool {
		// Only handle events for our window
		// Note: MouseEvent doesn't have WindowID in the current structure,
		// but since we're using a single window, we can assume all events are for this window
		if event.Action == types.MousePress {
			bbox, fixed, _ := bboxState.Get()

			// Check if click is within bounding box
			if bbox != nil && event.X >= bbox.Min.X && event.X <= bbox.Max.X &&
				event.Y >= bbox.Min.Y && event.Y <= bbox.Max.Y {
				if event.Button == types.MouseButtonLeft {
					// Left-click: Fix bounding box (keep current angle from state)
					if !fixed {
						_, angle, _, _ := bboxState.GetWithAngle()
						bboxState.SetFixedWithAngle(bbox, angle)
						slog.Info("Bounding box fixed via mouse click",
							"x", event.X, "y", event.Y,
							"bbox", bbox, "angle", angle)
					}
				} else if event.Button == types.MouseButtonRight {
					// Right-click: Toggle flip
					bboxState.ToggleFlip()
					slog.Info("Bounding box flip toggled via right-click",
						"x", event.X, "y", event.Y)
				}
			}
		}
		return true // Continue processing
	}

	// Keyboard event handler
	onKey := func(event types.KeyEvent) bool {
		if event.Action == types.KeyPress {
			switch event.Key {
			case 'f', 'F':
				// 'f' or 'F': Fix bounding box (keep current angle from state)
				bbox, angle, fixed, _ := bboxState.GetWithAngle()
				if bbox != nil && !fixed {
					bboxState.SetFixedWithAngle(bbox, angle)
					slog.Info("Bounding box fixed via keyboard ('f' key)", "bbox", bbox, "angle", angle)
				}
			case 'r', 'R':
				// 'r' or 'R': Reset (unfix) bounding box
				_, fixed, _ := bboxState.Get()
				if fixed {
					bboxState.Reset()
					slog.Info("Bounding box reset via keyboard ('r' key)")
				}
			case 'x', 'X':
				// 'x' or 'X': Toggle flip
				bboxState.ToggleFlip()
				slog.Info("Bounding box flip toggled via keyboard ('x' key)")
			case 27: // ESC key
				slog.Info("ESC key pressed, cancelling context")
				// Return false to stop event processing - display sink will call cancel function
				return false // Stop processing
			}
		}
		return true // Continue processing
	}

	// Close callback
	onClose := func(window *cv.Window, remainingWindows int) bool {
		slog.Info("Calibration window closed")
		if remainingWindows == 0 {
			return true // Cancel context when last window closes
		}
		return false
	}

	// Create camera display sink with mouse and keyboard handlers
	// Pass cancel function so ESC key can cancel the context
	cameraDisplaySink, err := gocv.NewDisplaySinkFromOptions(
		calibCtx,
		title,
		width,
		height,
		gocv.WithCancel(cancel),      // Cancel function called when handler returns false
		gocv.WithOnMouse(onMouse),
		gocv.WithOnKey(onKey),
		gocv.WithOnClose(onClose),
		types.WithRelease(), // Release tensors after displaying
	)
	if err != nil {
		return fmt.Errorf("failed to create camera display sink: %w", err)
	}
	defer cameraDisplaySink.Close()

	// Create spectrum display sink (second window)
	spectrumDisplaySink, err := gocv.NewDisplaySinkFromOptions(
		calibCtx,
		"Spectrum",
		800, // Default width for spectrum window
		600, // Default height for spectrum window
		gocv.WithCancel(cancel), // Same cancel function
		types.WithRelease(),      // Release tensors after displaying
	)
	if err != nil {
		return fmt.Errorf("failed to create spectrum display sink: %w", err)
	}
	defer spectrumDisplaySink.Close()

	// Continuous loop: capture, detect, display
	frameIndex := 0
	for {
		select {
		case <-calibCtx.Done():
			return calibCtx.Err()
		default:
			// Capture a frame
			frame, err := src.ReadFrame()
			if err != nil {
				return fmt.Errorf("failed to read frame: %w", err)
			}

			// Extract gocv Mat from frame
			mat, err := extractMatFromFrame(frame)
			if err != nil {
				slog.Warn("Failed to extract image from frame", "error", err)
				continue
			}

			// Get current bounding box state
			bbox, fixed, flipped := bboxState.Get()

			// Detect strip bounding box if not fixed
			if !fixed {
				detectedBbox, angle, err := detectStripBoundingBox(&mat)
				if err != nil {
					slog.Debug("Failed to detect strip", "error", err)
				} else if detectedBbox != nil {
					bboxState.SetDetected(detectedBbox, angle)
					bbox = detectedBbox
				}
			}

			// Save config when bounding box is fixed (first time only)
			// Use atomic compare-and-swap to ensure only one save occurs
			if fixed && shouldSaveConfig && !configSaved.Load() {
				// Update config with current bounding box, rotation angle, and flip state
				bbox, angle, _, flipped := bboxState.GetWithAngle()
				if bbox != nil {
					cfg.Window = rectangleToSpectrumWindow(bbox, angle, flipped)

					// Save config
					saver := config.NewSaver("")
					if err := saver.Save(calibCtx, configOutputPath, cfg); err != nil {
						slog.Error("Failed to save config", "error", err, "path", configOutputPath)
					} else {
						slog.Info("Config saved", "path", configOutputPath, "flipped", flipped)
						configSaved.Store(true)
					}
				}
			}

			// Draw bounding box with color coding
			if bbox != nil {
				// Red if not fixed, green if fixed
				drawBoundingBoxWithState(&mat, bbox, fixed, flipped)
			}

			// Create frame with annotated image for camera window
			annotatedFrame := createFrameFromMat(&mat, frameIndex)
			frameIndex++

			// Display the camera frame
			if err := cameraDisplaySink.WriteFrame(annotatedFrame); err != nil {
				slog.Warn("Failed to display camera frame", "error", err)
			}

			// Extract and display spectrum if bounding box is available
			if bbox != nil {
				// Get angle from state
				_, angle, _, _ := bboxState.GetWithAngle()

				// Extract spectrum from bounding box
				spd, err := extractSpectrumFromBoundingBox(&mat, bbox, angle, flipped)
				if err != nil {
					slog.Debug("Failed to extract spectrum", "error", err)
				} else {
					// Create spectrum visualization
					spectrumMat := cv.NewMatWithSize(600, 800, cv.MatTypeCV8UC3)
					defer spectrumMat.Close()

					// Render spectrum with peaks and calibration targets
					renderer := render.NewRenderer()
					// Get calibration target (default to Hg for now)
					target := GetCalibrationTarget("hg")
					if err := renderer.RenderSpectrumWithTargets(calibCtx, spectrumMat, spd, cfg.Display, target.Wavelengths, true); err != nil {
						slog.Debug("Failed to render spectrum", "error", err)
					} else {
						// Create frame with spectrum
						spectrumFrame := createFrameFromMat(&spectrumMat, frameIndex)

						// Display the spectrum frame
						if err := spectrumDisplaySink.WriteFrame(spectrumFrame); err != nil {
							slog.Debug("Failed to display spectrum frame", "error", err)
						}
					}
				}
			}

			mat.Close()
		}
	}
}

// extractMatFromFrame extracts gocv.Mat from types.Frame
func extractMatFromFrame(frame types.Frame) (cv.Mat, error) {
	if len(frame.Tensors) == 0 {
		return cv.Mat{}, fmt.Errorf("frame has no tensors")
	}

	// Try to extract Mat from first tensor
	tensor := frame.Tensors[0]

	// Check if tensor is a gocv tensor (Accessor interface)
	if gocvTensor, ok := tensor.(tensorgocv.Accessor); ok {
		// Clone the mat so we can modify it
		mat, err := gocvTensor.MatClone()
		if err != nil {
			return cv.Mat{}, fmt.Errorf("failed to clone mat: %w", err)
		}
		return mat, nil
	}

	return cv.Mat{}, fmt.Errorf("frame tensor is not a gocv tensor")
}

// RotatedBoundingBox represents a rotated bounding box with angle information
type RotatedBoundingBox struct {
	Rect  image.Rectangle
	Angle float64 // Rotation angle in degrees (-90 to +90)
}

// detectStripBoundingBox detects the spectral strip bounding box in the image
// Enhanced algorithm supports rotated strips (±45 degrees) and improved accuracy
// Returns the extended axis-aligned bounding box and the rotation angle
func detectStripBoundingBox(img *cv.Mat) (*image.Rectangle, float64, error) {
	if img.Empty() {
		return nil, 0.0, fmt.Errorf("image is empty")
	}

	// Convert to grayscale for processing
	gray := cv.NewMat()
	defer gray.Close()
	cv.CvtColor(*img, &gray, cv.ColorBGRToGray)

	// Apply adaptive thresholding for better detection in varying lighting
	thresh := cv.NewMat()
	defer thresh.Close()
	cv.AdaptiveThreshold(gray, &thresh, 255, cv.AdaptiveThresholdMean, cv.ThresholdBinary, 11, 10)

	// Apply morphological operations to clean up the binary image
	kernel := cv.GetStructuringElement(cv.MorphRect, image.Pt(3, 3))
	defer kernel.Close()

	cleaned := cv.NewMat()
	defer cleaned.Close()
	cv.MorphologyEx(thresh, &cleaned, cv.MorphClose, kernel)

	// Find contours
	contours := cv.FindContours(cleaned, cv.RetrievalExternal, cv.ChainApproxSimple)
	defer contours.Close()

	// Filter contours by size, aspect ratio, and orientation
	var bestCandidate *RotatedBoundingBox
	maxScore := 0.0

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)

		// Get minimum area rotated rectangle
		rotatedRect := cv.MinAreaRect(contour)

		// Extract angle (OpenCV returns angle from -90 to 0)
		angle := rotatedRect.Angle

		// Normalize angle to -90 to +90 range and handle special cases
		// MinAreaRect angle is negative when height > width
		width := rotatedRect.Width
		height := rotatedRect.Height

		// Swap if height > width (means angle is actually for the taller side)
		if height > width {
			width, height = height, width
			angle += 90.0
		}

		// Normalize angle to -45 to +45 range (strip should be roughly horizontal)
		for angle > 45 {
			angle -= 90
		}
		for angle < -45 {
			angle += 90
		}

		// Filter by angle: must be within ±45 degrees from horizontal
		if math.Abs(angle) > 45 {
			continue
		}

		// Calculate aspect ratio (strips are wide, not tall)
		// DIY spectrometers: typically ~100 pixels tall, ~1000+ pixels wide
		aspectRatio := float64(width) / float64(height)

		// Filter by aspect ratio: strips should have aspect ratio > 2.0 (wide)
		if aspectRatio < 2.0 {
			continue
		}

		// Calculate area
		area := width * height

		// Filter out very small detections
		if area < 1000 {
			continue
		}

		// Get image dimensions for width scoring
		imgSize := img.Size()
		imgWidth := imgSize[1]

		// Calculate score: prefer larger areas, better aspect ratios, and wider strips
		// Prioritize strips that extend across image width (typical for DIY spectrometers)
		score := float64(area) * aspectRatio

		// Bonus for strips that span more of the image width (horizontal strip should extend across frame)
		widthRatio := float64(width) / float64(imgWidth)
		widthBonus := 1.0 + (widthRatio * 0.5) // Up to 1.5x bonus for full-width strips
		score *= widthBonus

		// Prefer horizontal orientation (smaller angle deviation from 0)
		anglePenalty := 1.0 - (math.Abs(angle) / 45.0)
		score *= anglePenalty

		if score > maxScore {
			maxScore = score

			// Convert rotated rect to axis-aligned bounding box
			// Get the 4 corner points of the rotated rectangle
			points := rotatedRect.Points

			// Find min/max x and y from corner points
			if len(points) == 0 {
				continue
			}

			minX := points[0].X
			maxX := points[0].X
			minY := points[0].Y
			maxY := points[0].Y

			for _, pt := range points[1:] {
				if pt.X < minX {
					minX = pt.X
				}
				if pt.X > maxX {
					maxX = pt.X
				}
				if pt.Y < minY {
					minY = pt.Y
				}
				if pt.Y > maxY {
					maxY = pt.Y
				}
			}

			box := image.Rectangle{
				Min: image.Pt(minX, minY),
				Max: image.Pt(maxX, maxY),
			}

			bestCandidate = &RotatedBoundingBox{
				Rect:  box,
				Angle: angle,
			}
		}
	}

	if bestCandidate == nil {
		return nil, 0.0, nil
	}

	// Extend bounding box horizontally to full image width
	// The strip should extend across the entire image width (or as far as possible)
	imgSize := img.Size()
	imgWidth := imgSize[1]
	imgHeight := imgSize[0]

	// Get the detected bounding box
	detectedRect := bestCandidate.Rect

	// Calculate center Y and height for symmetric extension
	// Use center Y instead of Min.Y to properly center rotated strips
	centerY := (detectedRect.Min.Y + detectedRect.Max.Y) / 2
	height := detectedRect.Dy()

	// Extend horizontally: use center Y and extend symmetrically around center
	// The strip should span the full width of the image
	extendedRect := image.Rectangle{
		Min: image.Pt(0, centerY-height/2),        // Start at left edge, centered vertically
		Max: image.Pt(imgWidth, centerY+height/2), // End at right edge, centered vertically
	}

	// Clamp to image bounds (safety check)
	if extendedRect.Min.Y < 0 {
		extendedRect.Min.Y = 0
	}
	if extendedRect.Max.Y > imgHeight {
		extendedRect.Max.Y = imgHeight
	}
	if extendedRect.Min.X < 0 {
		extendedRect.Min.X = 0
	}
	if extendedRect.Max.X > imgWidth {
		extendedRect.Max.X = imgWidth
	}

	// Ensure valid rectangle
	if extendedRect.Dx() <= 0 || extendedRect.Dy() <= 0 {
		return nil, 0.0, nil
	}

	// Validate height reasonableness
	extendedHeight := extendedRect.Dy()
	maxReasonableHeight := imgHeight / 2 // 50% of image height
	if extendedHeight > maxReasonableHeight {
		slog.Warn("Extended bounding box height seems unreasonably large",
			"height", extendedHeight,
			"max_reasonable", maxReasonableHeight,
			"image_height", imgHeight,
			"ratio", float64(extendedHeight)/float64(imgHeight))
		// Continue anyway, but log warning
	}

	// Validate Y position margins (should not be at very top/bottom of image)
	marginThreshold := imgHeight / 20 // 5% margin threshold
	if extendedRect.Min.Y < marginThreshold {
		slog.Warn("Extended bounding box is very close to top edge",
			"y", extendedRect.Min.Y,
			"threshold", marginThreshold)
	}
	if extendedRect.Max.Y > imgHeight-marginThreshold {
		slog.Warn("Extended bounding box is very close to bottom edge",
			"y", extendedRect.Max.Y,
			"threshold", imgHeight-marginThreshold)
	}

	// Get the rotation angle from the detected strip
	rotationAngle := bestCandidate.Angle

	slog.Debug("Strip detected and extended",
		"detected_x", detectedRect.Min.X,
		"detected_y", detectedRect.Min.Y,
		"detected_width", detectedRect.Dx(),
		"detected_height", detectedRect.Dy(),
		"extended_x", extendedRect.Min.X,
		"extended_y", extendedRect.Min.Y,
		"extended_width", extendedRect.Dx(),
		"extended_height", extendedRect.Dy(),
		"angle", rotationAngle,
		"score", maxScore,
	)

	return &extendedRect, rotationAngle, nil
}

// drawBoundingBoxWithState draws a bounding box with color coding and state info
func drawBoundingBoxWithState(img *cv.Mat, bbox *image.Rectangle, fixed bool, flipped bool) {
	if bbox == nil {
		return
	}

	// Color coding: red if not fixed, green if fixed
	var boxColor color.RGBA
	if fixed {
		boxColor = color.RGBA{0, 255, 0, 255} // Green for fixed
	} else {
		boxColor = color.RGBA{0, 0, 255, 255} // Red for unfixed
	}

	// Draw rectangle
	cv.Rectangle(img, *bbox, boxColor, 3)

	// Draw text label with state info
	status := "DETECTED"
	if fixed {
		status = "FIXED"
	}
	if flipped {
		status += " (FLIPPED)"
	}
	text := fmt.Sprintf("%s: %dx%d", status, bbox.Dx(), bbox.Dy())
	pt := image.Pt(bbox.Min.X, bbox.Min.Y-10)
	white := color.RGBA{255, 255, 255, 255}
	cv.PutText(img, text, pt, cv.FontHersheyPlain, 1.0, white, 2)

	// Draw instructions
	if !fixed {
		instruction := "Click to FIX bounding box | Right-click to FLIP"
		instPt := image.Pt(10, img.Size()[0]-10)
		cv.PutText(img, instruction, instPt, cv.FontHersheyPlain, 0.7, white, 1)
	}
}

// drawBoundingBox draws a bounding box on the image (legacy, for backward compatibility)
func drawBoundingBox(img *cv.Mat, bbox *image.Rectangle) {
	drawBoundingBoxWithState(img, bbox, false, false)
}

// createFrameFromMat creates a types.Frame from gocv.Mat
func createFrameFromMat(mat *cv.Mat, index int) types.Frame {
	// Convert Mat to tensor (share the mat, don't adopt since we'll close it separately)
	tensor, err := tensorgocv.FromMat(*mat, tensorgocv.WithSharedMat())
	if err != nil {
		// If conversion fails, create frame with empty tensors
		return types.Frame{
			Index:   index,
			Tensors: []types.Tensor{},
			Metadata: map[string]interface{}{
				"source": "camera_calibration",
				"error":  err.Error(),
			},
		}
	}

	// Create frame
	frame := types.Frame{
		Index:   index,
		Tensors: []types.Tensor{tensor},
		Metadata: map[string]interface{}{
			"source": "camera_calibration",
		},
	}

	return frame
}
