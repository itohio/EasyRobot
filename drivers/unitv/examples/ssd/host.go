//go:build !unitv

// The host side of the ssd example.
//
// With -gen it writes the board build's inputs (model.kgraph, input.i8,
// data_unitv.c) from a TFLite detection model and a raw test image:
//
//	go run ./drivers/unitv/examples/ssd -gen -model detect.tflite -image image256.u8
//
// Without it, it runs the same program as the board on the emulated KPU,
// from model.kgraph and input.i8, and prints the board's lines: its hashes
// and detections are the reference the board is compared with.
//
//	go run ./drivers/unitv/examples/ssd [-runs 3] [-dump]
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kgraph"
)

func main() {
	dir := "drivers/unitv/examples/ssd"
	gen := flag.Bool("gen", false, "write model.kgraph, input.i8 and data_unitv.c into -dir")
	modelPath := flag.String("model", filepath.Join(dir, "detect.tflite"), "with -gen: SSD-MobileNetV1 TFLite detection model")
	imagePath := flag.String("image", filepath.Join(dir, "input.u8"), "with -gen: 256×256×3 uint8 RGB image (raw)")
	out := flag.String("dir", dir, "directory of model.kgraph and input.i8")
	runs := flag.Int("runs", 3, "frames to report")
	dump := flag.Bool("dump", false, "also print the first frame's tensors")
	flag.Parse()
	var err error
	if *gen {
		err = generate(*modelPath, *imagePath, *out)
	} else {
		err = run(*out, *runs, *dump)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "ssd:", err)
		os.Exit(1)
	}
}

func run(dir string, runs int, dump bool) error {
	raw, err := os.ReadFile(filepath.Join(dir, "model.kgraph"))
	if err != nil {
		return err
	}
	model := kpu.AlignedCopy(raw) // the board links it at an aligned address
	img, err := os.ReadFile(filepath.Join(dir, "input.i8"))
	if err != nil {
		return err
	}
	in := unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(img))), len(img))
	fmt.Printf("ssd: boot model_addr_mod128 %d model_bytes %d input_bytes %d\r\n", uintptr(unsafe.Pointer(&model[0]))%128, len(model), len(in))
	t0 := time.Now()
	f, err := kgraph.Parse(model)
	if err != nil {
		return err
	}
	fmt.Printf("ssd: parse ok ms %d\r\n", time.Since(t0).Milliseconds())
	d, err := newDetector(f, kpu.NewEmulated)
	if err != nil {
		return err
	}
	return loop(os.Stdout, d, in, time.Since(t0), 0, runs, dump)
}
