package gocv

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

func TestFrameStreamFromGlob(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "frame1.png"),
		filepath.Join(dir, "frame2.png"),
	}

	for idx, path := range paths {
		if err := writeTestImage(path, color.RGBA{R: uint8(idx * 50), G: uint8(100 + idx*20), B: uint8(150 - idx*30), A: 255}); err != nil {
			t.Fatalf("failed to write test image: %v", err)
		}
	}

	pattern := filepath.Join(dir, "*.png")

	unmarshaller := NewUnmarshaller(WithPath(pattern))

	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
		t.Fatalf("unmarshal stream: %v", err)
	}
	defer stream.Close()

	var (
		count int
		seen  = map[string]bool{}
	)

	for frame := range stream.C {
		if frame.Index != count {
			t.Fatalf("expected frame index %d, got %d", count, frame.Index)
		}
		if frame.Timestamp <= 0 {
			t.Fatalf("expected positive timestamp, got %d", frame.Timestamp)
		}
		if len(frame.Tensors) == 0 {
			t.Fatalf("frame %d has no tensors", count)
		}

		metaPath, ok := frame.Metadata["path"].(string)
		if !ok || metaPath == "" {
			t.Fatalf("frame %d missing path metadata: %v", count, frame.Metadata)
		}
		seen[metaPath] = true

		for _, tensor := range frame.Tensors {
			tensor.Release()
		}
		count++
	}

	if count != len(paths) {
		t.Fatalf("expected %d frames, got %d", len(paths), count)
	}

	for _, path := range paths {
		if !seen[path] {
			t.Fatalf("expected path %s in stream output", path)
		}
	}
}

func TestFrameStreamFromMultipleGlobs(t *testing.T) {
	pattern1 := filepath.Join("..", "testdata", "img1", "*.png")
	pattern2 := filepath.Join("..", "testdata", "img2", "*.png")

	files1, err := filepath.Glob(pattern1)
	if err != nil || len(files1) == 0 {
		t.Fatalf("glob %s: %v", pattern1, err)
	}
	files2, err := filepath.Glob(pattern2)
	if err != nil || len(files2) == 0 {
		t.Fatalf("glob %s: %v", pattern2, err)
	}
	if len(files1) != len(files2) {
		t.Fatalf("expected equal number of files in img1 and img2, got %d vs %d", len(files1), len(files2))
	}

	unmarshaller := NewUnmarshaller(
		WithPath(pattern1),
		WithPath(pattern2),
	)

	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(bytes.NewReader(nil), &stream); err != nil {
		t.Fatalf("unmarshal stream: %v", err)
	}
	defer stream.Close()

	count := 0
	for frame := range stream.C {
		if frame.Index != count {
			t.Fatalf("expected frame index %d, got %d", count, frame.Index)
		}
		if frame.Timestamp <= 0 {
			t.Fatalf("expected positive timestamp, got %d", frame.Timestamp)
		}
		if len(frame.Tensors) != 2 {
			t.Fatalf("expected 2 tensors per frame, got %d", len(frame.Tensors))
		}

		sources, ok := frame.Metadata["sources"].([]map[string]any)
		if !ok || len(sources) != 2 {
			t.Fatalf("expected metadata.sources length 2, got %v", frame.Metadata["sources"])
		}

		path1, _ := sources[0]["path"].(string)
		path2, _ := sources[1]["path"].(string)

		if path1 != files1[count] {
			t.Fatalf("frame %d expected path1 %s, got %s", count, files1[count], path1)
		}
		if path2 != files2[count] {
			t.Fatalf("frame %d expected path2 %s, got %s", count, files2[count], path2)
		}

		for _, tensor := range frame.Tensors {
			tensor.Release()
		}
		count++
	}

	if count != len(files1) {
		t.Fatalf("expected %d frames, got %d", len(files1), count)
	}
}

func writeTestImage(path string, c color.Color) error {
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			img.Set(x, y, c)
		}
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return png.Encode(file, img)
}
