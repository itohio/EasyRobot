package source

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log/slog"
	"math"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

var (
	generateWidth    int
	generateHeight   int
	generateFPS      int
	generateDuration time.Duration
)

// generatorSource implements Source for generating test frames with a spinning triangle.
type generatorSource struct {
	baseSource
	width     int
	height    int
	fps       int
	duration  time.Duration
	frameCh   chan types.Frame
	startTime time.Time
	frameIdx  int
}

// NewGeneratorSource creates a new generator source.
func NewGeneratorSource() Source {
	return &generatorSource{}
}

func (s *generatorSource) RegisterFlags() {
	flag.IntVar(&generateWidth, "generate-width", 640, "Width of generated frames")
	flag.IntVar(&generateHeight, "generate-height", 480, "Height of generated frames")
	flag.IntVar(&generateFPS, "generate-fps", 30, "Frames per second for generation")
	flag.DurationVar(&generateDuration, "generate-duration", 0, "Duration to generate frames (0 = infinite)")
}

func (s *generatorSource) Start(ctx context.Context) error {
	s.width = generateWidth
	s.height = generateHeight
	s.fps = generateFPS
	s.duration = generateDuration
	s.startTime = time.Now()
	s.frameIdx = 0

	// Validate and set defaults
	if s.fps <= 0 {
		slog.Warn("Invalid or zero FPS, using default 30", "fps", s.fps)
		s.fps = 30
	}
	if s.width <= 0 {
		slog.Warn("Invalid or zero width, using default 640", "width", s.width)
		s.width = 640
	}
	if s.height <= 0 {
		slog.Warn("Invalid or zero height, using default 480", "height", s.height)
		s.height = 480
	}

	slog.Info("Starting generator source",
		"width", s.width,
		"height", s.height,
		"fps", s.fps,
		"duration", s.duration,
	)

	// Create frame channel
	s.frameCh = make(chan types.Frame, 10)

	// Start generator goroutine
	go s.generateFrames(ctx)

	// Create FrameStream
	s.stream = types.NewFrameStream(s.frameCh, func() {
		close(s.frameCh)
	})

	return s.baseSource.Start(ctx)
}

func (s *generatorSource) generateFrames(ctx context.Context) {
	defer close(s.frameCh)

	if s.fps <= 0 {
		slog.Error("Cannot generate frames with zero or negative FPS", "fps", s.fps)
		return
	}

	frameInterval := time.Second / time.Duration(s.fps)
	slog.Info("Generator frame loop started",
		"fps", s.fps,
		"frame_interval", frameInterval,
		"duration", s.duration,
	)

	ticker := time.NewTicker(frameInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("Generator stopping", "reason", "context cancelled", "frames_generated", s.frameIdx)
			return
		case <-ticker.C:
			// Check duration limit
			elapsed := time.Since(s.startTime)
			if s.duration > 0 && elapsed >= s.duration {
				slog.Info("Generator stopping", "reason", "duration reached", "frames_generated", s.frameIdx, "elapsed", elapsed)
				return
			}

			// Generate frame
			slog.Debug("Generating frame", "frame_index", s.frameIdx, "elapsed", elapsed)
			frame, err := s.createFrame()
			if err != nil {
				slog.Error("Failed to create frame", "frame_index", s.frameIdx, "err", err)
				// Send error frame
				frame = types.Frame{
					Index:    s.frameIdx,
					Metadata: map[string]any{"error": err},
				}
			} else {
				slog.Debug("Frame created successfully", "frame_index", s.frameIdx, "timestamp", frame.Timestamp)
			}

			select {
			case <-ctx.Done():
				slog.Info("Generator stopping", "reason", "context cancelled during send", "frames_generated", s.frameIdx)
				return
			case s.frameCh <- frame:
				s.frameIdx++
				if s.frameIdx%100 == 0 {
					slog.Info("Generator progress", "frames_generated", s.frameIdx, "elapsed", elapsed)
				}
			}
		}
	}
}

func (s *generatorSource) createFrame() (types.Frame, error) {
	slog.Debug("Creating frame", "width", s.width, "height", s.height, "frame_index", s.frameIdx)

	// Create RGBA image using standard library
	img := image.NewRGBA(image.Rect(0, 0, s.width, s.height))

	// Fill with black background
	for y := 0; y < s.height; y++ {
		for x := 0; x < s.width; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}
	slog.Debug("Image created and filled", "size", s.width*s.height)

	// Calculate rotation angle based on frame index
	angle := float64(s.frameIdx) * 2.0 * math.Pi / float64(s.fps) // One full rotation per second

	// Calculate triangle center
	centerX := float64(s.width) / 2.0
	centerY := float64(s.height) / 2.0

	// Triangle size (radius from center to vertex)
	triangleSize := math.Min(float64(s.width), float64(s.height)) * 0.3

	// Calculate triangle vertices (equilateral triangle)
	vertices := make([]image.Point, 3)
	for i := 0; i < 3; i++ {
		// Start with triangle pointing up, then rotate
		vertexAngle := angle + float64(i)*2.0*math.Pi/3.0
		x := centerX + triangleSize*math.Cos(vertexAngle)
		y := centerY + triangleSize*math.Sin(vertexAngle)
		vertices[i] = image.Point{
			X: int(x),
			Y: int(y),
		}
	}

	// Draw filled triangle using scanline fill
	s.drawFilledTriangle(img, vertices, color.RGBA{0, 255, 0, 255})

	// Draw triangle outline
	green := color.RGBA{0, 200, 0, 255}
	s.drawLine(img, vertices[0], vertices[1], green)
	s.drawLine(img, vertices[1], vertices[2], green)
	s.drawLine(img, vertices[2], vertices[0], green)

	// Convert image.RGBA to gocv Mat
	slog.Debug("Converting image to gocv Mat")
	mat, err := cv.ImageToMatRGB(img)
	if err != nil {
		return types.Frame{}, fmt.Errorf("failed to convert image to mat: %w", err)
	}
	slog.Debug("Image converted to Mat", "mat_empty", mat.Empty(), "mat_rows", mat.Rows(), "mat_cols", mat.Cols())

	// Convert Mat to Tensor (tensor will own the mat, so don't close it here)
	slog.Debug("Converting Mat to Tensor")
	tensor, err := gocv.FromMat(mat, gocv.WithAdoptedMat())
	if err != nil {
		mat.Close()
		return types.Frame{}, fmt.Errorf("failed to create tensor from mat: %w", err)
	}
	slog.Debug("Mat converted to Tensor", "tensor_shape", tensor.Shape())

	// Create frame
	frame := types.Frame{
		Index:     s.frameIdx,
		Timestamp: time.Since(s.startTime).Nanoseconds(),
		Metadata: map[string]any{
			"source":      "generator",
			"width":       s.width,
			"height":      s.height,
			"angle":       angle,
			"frame_index": s.frameIdx,
		},
		Tensors: []types.Tensor{tensor},
	}

	return frame, nil
}

// drawLine draws a line between two points using Bresenham's algorithm
func (s *generatorSource) drawLine(img *image.RGBA, p1, p2 image.Point, c color.RGBA) {
	dx := abs(p2.X - p1.X)
	dy := abs(p2.Y - p1.Y)
	sx := 1
	if p1.X > p2.X {
		sx = -1
	}
	sy := 1
	if p1.Y > p2.Y {
		sy = -1
	}
	err := dx - dy

	x, y := p1.X, p1.Y
	for {
		if x >= 0 && x < s.width && y >= 0 && y < s.height {
			img.Set(x, y, c)
		}
		if x == p2.X && y == p2.Y {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}
}

// drawFilledTriangle draws a filled triangle
func (s *generatorSource) drawFilledTriangle(img *image.RGBA, vertices []image.Point, c color.RGBA) {
	// Find bounding box
	minX, minY := vertices[0].X, vertices[0].Y
	maxX, maxY := vertices[0].X, vertices[0].Y
	for _, v := range vertices {
		if v.X < minX {
			minX = v.X
		}
		if v.X > maxX {
			maxX = v.X
		}
		if v.Y < minY {
			minY = v.Y
		}
		if v.Y > maxY {
			maxY = v.Y
		}
	}

	// Clamp to image bounds
	if minX < 0 {
		minX = 0
	}
	if minY < 0 {
		minY = 0
	}
	if maxX >= s.width {
		maxX = s.width - 1
	}
	if maxY >= s.height {
		maxY = s.height - 1
	}

	// Fill pixels inside triangle
	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			if s.pointInTriangle(x, y, vertices) {
				img.Set(x, y, c)
			}
		}
	}
}

// pointInTriangle checks if a point is inside a triangle using barycentric coordinates
func (s *generatorSource) pointInTriangle(px, py int, vertices []image.Point) bool {
	v0x, v0y := float64(vertices[2].X-vertices[0].X), float64(vertices[2].Y-vertices[0].Y)
	v1x, v1y := float64(vertices[1].X-vertices[0].X), float64(vertices[1].Y-vertices[0].Y)
	v2x, v2y := float64(px-vertices[0].X), float64(py-vertices[0].Y)

	dot00 := v0x*v0x + v0y*v0y
	dot01 := v0x*v1x + v0y*v1y
	dot02 := v0x*v2x + v0y*v2y
	dot11 := v1x*v1x + v1y*v1y
	dot12 := v1x*v2x + v1y*v2y

	invDenom := 1 / (dot00*dot11 - dot01*dot01)
	u := (dot11*dot02 - dot01*dot12) * invDenom
	v := (dot00*dot12 - dot01*dot02) * invDenom

	return u >= 0 && v >= 0 && u+v <= 1
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
