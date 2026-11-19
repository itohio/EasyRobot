//go:build unitv

// Command yolo runs YOLOX-nano (int8, 416×416, COCO) on the UnitV's CPU
// with qnet's kernels on one built-in test image and prints, every run, the
// frame time, FNV-1a hashes of both output tensors, heap figures, the
// detections and the time per layer kind on UARTHS (USB serial, 115200
// baud). The same source runs on the host (host.go) for the reference
// output; the board's lines must equal it except for times and heap. The
// model and the int8 input are the constants model and input of
// data_unitv.go, written by gen.go (README.md).
package main

import (
	"os"
	"time"
	"unsafe"
)

func main() {
	if err := run(); err != nil {
		for {
			println("yolo: error", err.Error())
			time.Sleep(2 * time.Second)
		}
	}
}

func run() error {
	t0 := time.Now()
	// The loader aliases the int8 weights and Run only reads its input:
	// both constants are used in place, the only copies.
	d, err := newDetector(unsafe.Slice(unsafe.StringData(model), len(model)))
	if err != nil {
		return err
	}
	setup := time.Since(t0)
	in := unsafe.Slice((*int8)(unsafe.Pointer(unsafe.StringData(input))), len(input))
	return loop(os.Stdout, d, in, setup, 0)
}
