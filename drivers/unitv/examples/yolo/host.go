//go:build !unitv

// The host build of the same program: it reads the model and the image
// from files (no embedding, so the package builds without them) and prints
// the board's lines for a fixed number of runs; its hashes and detections
// are the reference the board must reproduce.
//
//	go run ./drivers/unitv/examples/yolo -model model.tflite -input input.u8
package main

import (
	"flag"
	"fmt"
	"os"
	"time"
)

func main() {
	modelPath := flag.String("model", "drivers/unitv/examples/yolo/model.tflite", "YOLOX-nano int8 TFLite file")
	inputPath := flag.String("input", "drivers/unitv/examples/yolo/input.u8", "416×416×3 uint8 RGB image (raw)")
	runs := flag.Int("runs", 10, "runs to report")
	flag.Parse()
	if err := run(*modelPath, *inputPath, *runs); err != nil {
		fmt.Fprintln(os.Stderr, "yolo:", err)
		os.Exit(1)
	}
}

func run(modelPath, inputPath string, runs int) error {
	model, err := os.ReadFile(modelPath)
	if err != nil {
		return err
	}
	image, err := os.ReadFile(inputPath)
	if err != nil {
		return err
	}
	if len(image) != inputSize*inputSize*3 {
		return fmt.Errorf("%s: %d bytes, want %d (416×416×3)", inputPath, len(image), inputSize*inputSize*3)
	}
	t0 := time.Now()
	d, err := newDetector(model)
	if err != nil {
		return err
	}
	setup := time.Since(t0)
	in := make([]int8, len(image))
	toInt8(in, image)
	return loop(os.Stdout, d, in, setup, runs)
}

// toInt8 converts a uint8 RGB image into the graph's int8 input (the cut
// QUANTIZE: uint8 (1, 0) → int8 (1, −128)).
func toInt8(dst []int8, src []byte) {
	for i, v := range src {
		dst[i] = int8(int(v) - 128)
	}
}
