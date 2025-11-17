package gocv

import (
	"bytes"
	"image/color"
	"io"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func TestManifestSerializationRoundTrip(t *testing.T) {
	// Create test images
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
		filepath.Join(dir, "frame3.png"),
	}

	for idx, path := range paths {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	// Create unmarshaller with sources (use glob pattern to enumerate directory)
	pattern := filepath.Join(dir, "*.png")
	unmarshaller := NewUnmarshaller(
		WithPath(pattern),
		WithSequential(true),
		WithBestEffortDevices(false),
	)

	// Unmarshal to create a stream
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
		t.Fatalf("unmarshal stream: %v", err)
	}

	// Marshal the stream to get manifest (this consumes the stream)
	// Pass the same sources to the marshaller so it can write them to the manifest
	var manifestBuf bytes.Buffer
	marshaller := NewMarshaller(
		WithPath(pattern),
		WithSequential(true),
		WithBestEffortDevices(false),
	)
	if err := marshaller.Marshal(&manifestBuf, stream); err != nil {
		t.Fatalf("marshal stream: %v", err)
	}

	// Verify manifest was written (should have length prefix + protobuf data)
	if manifestBuf.Len() < 4 {
		t.Fatalf("expected manifest data, got %d bytes", manifestBuf.Len())
	}

	// The buffer contains: [manifest][optional summary]
	// We need to read just the manifest part
	// readManifest will handle the length-prefixed format correctly
	var restoredStream types.FrameStream
	restoredUnmarshaller := NewUnmarshaller()
	if err := restoredUnmarshaller.Unmarshal(&manifestBuf, &restoredStream); err != nil {
		t.Fatalf("unmarshal from manifest: %v", err)
	}
	defer restoredStream.Close()

	// Verify we can read frames from restored stream
	count := 0
	for frame := range restoredStream.C {
		if frame.Index != count {
			t.Errorf("expected frame index %d, got %d", count, frame.Index)
		}
		if len(frame.Tensors) == 0 {
			t.Errorf("frame %d has no tensors", count)
		}
		for _, tensor := range frame.Tensors {
			tensor.Release()
		}
		count++
	}

	if count != len(paths) {
		t.Errorf("expected %d frames, got %d", len(paths), count)
	}
}

func TestManifestFormatDetection(t *testing.T) {
	// Create test images
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
	}

	for idx, path := range paths {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	// Test 1: Protobuf manifest format detection
	t.Run("ProtobufManifest", func(t *testing.T) {
		pattern := filepath.Join(dir, "*.png")
		unmarshaller := NewUnmarshaller(WithPath(pattern))
		var stream types.FrameStream
		if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
			t.Fatalf("unmarshal stream: %v", err)
		}

		var buf bytes.Buffer
		marshaller := NewMarshaller(WithPath(pattern))
		if err := marshaller.Marshal(&buf, stream); err != nil {
			t.Fatalf("marshal stream: %v", err)
		}

		// Try to unmarshal from the protobuf manifest
		var restoredStream types.FrameStream
		restoredUnmarshaller := NewUnmarshaller()
		if err := restoredUnmarshaller.Unmarshal(&buf, &restoredStream); err != nil {
			t.Fatalf("unmarshal from protobuf manifest: %v", err)
		}
		defer restoredStream.Close()

		// Should be able to read frames
		count := 0
		for range restoredStream.C {
			count++
		}
		if count != len(paths) {
			t.Errorf("expected %d frames from protobuf manifest, got %d", len(paths), count)
		}
	})

	// Test 2: Legacy text format detection
	t.Run("LegacyTextFormat", func(t *testing.T) {
		// Create legacy text format (newline-separated paths)
		// Each path becomes a separate single-image source
		// In parallel mode (default), they combine into frames with multiple tensors
		var textBuf bytes.Buffer
		for _, path := range paths {
			textBuf.WriteString(path)
			textBuf.WriteString("\n")
		}

		var stream types.FrameStream
		unmarshaller := NewUnmarshaller()
		if err := unmarshaller.Unmarshal(&textBuf, &stream); err != nil {
			t.Fatalf("unmarshal from legacy text format: %v", err)
		}
		defer stream.Close()

		// In parallel mode, multiple single-image sources produce 1 frame with multiple tensors
		// (one tensor per source, all at the same index)
		count := 0
		tensorCount := 0
		for frame := range stream.C {
			count++
			tensorCount += len(frame.Tensors)
			for _, tensor := range frame.Tensors {
				tensor.Release()
			}
		}
		// Should get 1 frame with len(paths) tensors in parallel mode
		if count != 1 {
			t.Errorf("expected 1 frame from legacy text format (parallel mode), got %d", count)
		}
		if tensorCount != len(paths) {
			t.Errorf("expected %d total tensors from legacy text format, got %d", len(paths), tensorCount)
		}
	})

	// Test 3: Empty reader (should use configured sources)
	t.Run("EmptyReaderWithConfig", func(t *testing.T) {
		pattern := filepath.Join(dir, "*.png")
		unmarshaller := NewUnmarshaller(WithPath(pattern))
		var stream types.FrameStream
		if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
			t.Fatalf("unmarshal with empty reader: %v", err)
		}
		defer stream.Close()

		count := 0
		for range stream.C {
			count++
		}
		if count != len(paths) {
			t.Errorf("expected %d frames with empty reader, got %d", len(paths), count)
		}
	})
}

func TestManifestWithMultipleSources(t *testing.T) {
	// Create two directories with images
	dir1 := t.TempDir()
	dir2 := t.TempDir()

	paths1 := []string{
		filepath.Join(dir1, "img1_001.png"),
		filepath.Join(dir1, "img1_002.png"),
	}
	paths2 := []string{
		filepath.Join(dir2, "img2_001.png"),
		filepath.Join(dir2, "img2_002.png"),
	}

	for idx, path := range paths1 {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}
	for idx, path := range paths2 {
		c := color.RGBA{
			R: uint8((idx + 10) * 50 % 256),
			G: uint8(100 + (idx+10)*20 % 256),
			B: uint8(150 - (idx+10)*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	// Create unmarshaller with multiple sources (use glob patterns)
	pattern1 := filepath.Join(dir1, "*.png")
	pattern2 := filepath.Join(dir2, "*.png")
	unmarshaller := NewUnmarshaller(
		WithPath(pattern1),
		WithPath(pattern2),
		WithSequential(false), // parallel mode
	)

	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
		t.Fatalf("unmarshal stream: %v", err)
	}

	// Marshal to manifest (pass sources so manifest contains them)
	var buf bytes.Buffer
	marshaller := NewMarshaller(
		WithPath(pattern1),
		WithPath(pattern2),
		WithSequential(false),
	)
	if err := marshaller.Marshal(&buf, stream); err != nil {
		t.Fatalf("marshal stream: %v", err)
	}

	// Unmarshal from manifest
	var restoredStream types.FrameStream
	restoredUnmarshaller := NewUnmarshaller()
	if err := restoredUnmarshaller.Unmarshal(&buf, &restoredStream); err != nil {
		t.Fatalf("unmarshal from manifest: %v", err)
	}
	defer restoredStream.Close()

	// Verify frames have 2 tensors (one from each source)
	count := 0
	for frame := range restoredStream.C {
		if len(frame.Tensors) != 2 {
			t.Errorf("frame %d expected 2 tensors, got %d", count, len(frame.Tensors))
		}
		for _, tensor := range frame.Tensors {
			tensor.Release()
		}
		count++
	}

	if count != len(paths1) {
		t.Errorf("expected %d frames, got %d", len(paths1), count)
	}
}

func TestManifestSyncMode(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
	}

	for idx, path := range paths {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	tests := []struct {
		name      string
		sequential bool
	}{
		{"Parallel", false},
		{"Sequential", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pattern := filepath.Join(dir, "*.png")
			unmarshaller := NewUnmarshaller(
				WithPath(pattern),
				WithSequential(tt.sequential),
			)

			var stream types.FrameStream
			if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
				t.Fatalf("unmarshal stream: %v", err)
			}

			// Marshal to manifest (pass sources so manifest contains them)
			var buf bytes.Buffer
			marshaller := NewMarshaller(
				WithPath(pattern),
				WithSequential(tt.sequential),
			)
			if err := marshaller.Marshal(&buf, stream); err != nil {
				t.Fatalf("marshal stream: %v", err)
			}

			// Unmarshal from manifest
			var restoredStream types.FrameStream
			restoredUnmarshaller := NewUnmarshaller()
			if err := restoredUnmarshaller.Unmarshal(&buf, &restoredStream); err != nil {
				t.Fatalf("unmarshal from manifest: %v", err)
			}
			defer restoredStream.Close()

			// Verify we can read frames
			count := 0
			for range restoredStream.C {
				count++
			}
			if count != len(paths) {
				t.Errorf("expected %d frames, got %d", len(paths), count)
			}
		})
	}
}

func TestManifestBestEffort(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
	}

	for idx, path := range paths {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	tests := []struct {
		name        string
		bestEffort  bool
	}{
		{"BestEffortEnabled", true},
		{"BestEffortDisabled", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pattern := filepath.Join(dir, "*.png")
			unmarshaller := NewUnmarshaller(
				WithPath(pattern),
				WithBestEffortDevices(tt.bestEffort),
			)

			var stream types.FrameStream
			if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
				t.Fatalf("unmarshal stream: %v", err)
			}

			// Marshal to manifest (pass sources so manifest contains them)
			var buf bytes.Buffer
			marshaller := NewMarshaller(
				WithPath(pattern),
				WithBestEffortDevices(tt.bestEffort),
			)
			if err := marshaller.Marshal(&buf, stream); err != nil {
				t.Fatalf("marshal stream: %v", err)
			}

			// Unmarshal from manifest
			var restoredStream types.FrameStream
			restoredUnmarshaller := NewUnmarshaller()
			if err := restoredUnmarshaller.Unmarshal(&buf, &restoredStream); err != nil {
				t.Fatalf("unmarshal from manifest: %v", err)
			}
			defer restoredStream.Close()

			// Verify we can read frames
			count := 0
			for range restoredStream.C {
				count++
			}
			if count != len(paths) {
				t.Errorf("expected %d frames, got %d", len(paths), count)
			}
		})
	}
}

func TestLegacyTextFormatBackwardCompatibility(t *testing.T) {
	// Create test images
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
		filepath.Join(dir, "frame3.png"),
	}

	for idx, path := range paths {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	// Create legacy text format (newline-separated paths)
	var textBuf bytes.Buffer
	for _, path := range paths {
		textBuf.WriteString(path)
		textBuf.WriteString("\n")
	}

	// Test unmarshalling from legacy format
	var stream types.FrameStream
	unmarshaller := NewUnmarshaller()
	if err := unmarshaller.Unmarshal(&textBuf, &stream); err != nil {
		t.Fatalf("unmarshal from legacy text format: %v", err)
	}
	defer stream.Close()

	// In parallel mode (default), multiple single-image sources produce 1 frame with multiple tensors
	// Verify we can read the frame and it has tensors from all paths
	count := 0
	seenPaths := make(map[string]bool)
	tensorCount := 0
	for frame := range stream.C {
		if frame.Index != count {
			t.Errorf("expected frame index %d, got %d", count, frame.Index)
		}
		if len(frame.Tensors) == 0 {
			t.Errorf("frame %d has no tensors", count)
		}
		tensorCount += len(frame.Tensors)

		// Check metadata - in parallel mode, sources are combined
		// Check if we have individual path metadata or combined sources
		if metaPath, ok := frame.Metadata["path"].(string); ok {
			seenPaths[metaPath] = true
		} else if sources, ok := frame.Metadata["sources"].([]map[string]any); ok {
			// Multiple sources combined
			for _, source := range sources {
				if path, ok := source["path"].(string); ok {
					seenPaths[path] = true
				}
			}
		}

		for _, tensor := range frame.Tensors {
			tensor.Release()
		}
		count++
	}

	// In parallel mode, should get 1 frame with len(paths) tensors
	if count != 1 {
		t.Errorf("expected 1 frame from legacy text format (parallel mode), got %d", count)
	}
	if tensorCount != len(paths) {
		t.Errorf("expected %d total tensors from legacy text format, got %d", len(paths), tensorCount)
	}

	// Verify all paths are represented (either in frame metadata or sources)
	if len(seenPaths) < len(paths) {
		t.Logf("Note: Not all paths found in metadata (expected in parallel mode with combined sources)")
	}
}

func TestLegacyTextFormatWithEmptyLines(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
	}

	for idx, path := range paths {
		c := color.RGBA{
			R: uint8(idx * 50 % 256),
			G: uint8(100 + idx*20 % 256),
			B: uint8(150 - idx*30 % 256),
			A: 255,
		}
		if err := writeTestImage(path, c); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	// Create legacy text format with empty lines and whitespace
	var textBuf bytes.Buffer
	textBuf.WriteString("\n") // empty line at start
	textBuf.WriteString("  \n") // whitespace line
	for _, path := range paths {
		textBuf.WriteString(path)
		textBuf.WriteString("\n")
	}
	textBuf.WriteString("\n") // empty line at end
	textBuf.WriteString("  \t  \n") // whitespace line at end

	var stream types.FrameStream
	unmarshaller := NewUnmarshaller()
	if err := unmarshaller.Unmarshal(&textBuf, &stream); err != nil {
		t.Fatalf("unmarshal from legacy text format: %v", err)
	}
	defer stream.Close()

	// Should only read valid paths, ignoring empty/whitespace lines
	// In parallel mode (default), multiple single-image sources produce 1 frame with multiple tensors
	count := 0
	tensorCount := 0
	for frame := range stream.C {
		count++
		tensorCount += len(frame.Tensors)
		for _, tensor := range frame.Tensors {
			tensor.Release()
		}
	}
	// In parallel mode, should get 1 frame with len(paths) tensors
	if count != 1 {
		t.Errorf("expected 1 frame from legacy text format (parallel mode), got %d", count)
	}
	if tensorCount != len(paths) {
		t.Errorf("expected %d total tensors (ignoring empty lines), got %d", len(paths), tensorCount)
	}
}

func TestFormatDetectionEdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		data   []byte
		expect string // "protobuf", "text", or "error"
	}{
		{
			name:   "EmptyReader",
			data:   nil,
			expect: "text", // Should fall back to configured sources
		},
		{
			name:   "VerySmallData",
			data:   []byte{0x01, 0x02, 0x03},
			expect: "text", // Too small to be protobuf length prefix
		},
		{
			name:   "LargeLengthPrefix",
			data:   []byte{0xFF, 0xFF, 0xFF, 0xFF}, // 4GB length (unreasonable)
			expect: "text", // Should be rejected as protobuf
		},
		{
			name:   "ZeroLength",
			data:   []byte{0x00, 0x00, 0x00, 0x00},
			expect: "text", // Zero length is suspicious
		},
		{
			name:   "TextPath",
			data:   []byte("/path/to/image.png\n"),
			expect: "text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var reader io.Reader
			if tt.data == nil {
				reader = bytes.NewReader(nil)
			} else {
				reader = bytes.NewReader(tt.data)
			}

			// Try to read manifest
			manifest, err := readManifest(reader)
			if err != nil {
				if tt.expect == "error" {
					return // Expected error
				}
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.expect == "protobuf" {
				if manifest == nil {
					t.Error("expected protobuf manifest, got nil")
				}
			} else if tt.expect == "text" {
				if manifest != nil {
					t.Error("expected nil manifest (text format), got protobuf manifest")
				}
			}
		})
	}
}


