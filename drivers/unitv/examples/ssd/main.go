//go:build unitv

// Command ssd runs SSD-MobileNetV1 (uint8, COCO, at 256×256) on the UnitV:
// the convolutions on the K210's KPU, the small layers, the logistic and
// the box decoding on the CPU, on UARTHS (USB serial, 115200 baud).
//
// The model is a kgraph file (drivers/kpu/kgraph) that the host compiles
// (host.go, -gen) and data_unitv.c links into the image at a 128-byte-
// aligned address; nothing of it is copied or compiled on the board. The
// program runs on the test input linked beside it and prints, like the host
// build, the hash of every segment's output for the first frame, then the
// time per stage, the detections and the heap figures of every frame.
//
// Build-time option (tinygo -ldflags "-X main.mode=dump"): also print the
// first frame's tensors as hex, for comparing values with the host's.
package main

/*
extern const unsigned char ssd_model;
extern const unsigned char ssd_model_end;
extern const unsigned char ssd_input;
extern const unsigned char ssd_input_end;
*/
import "C"

import (
	"machine"
	"os"
	"runtime"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kgraph"
)

var mode = "run"

func main() {
	clearStack()
	err := run()
	for err != nil {
		println("ssd: error", err.Error())
		time.Sleep(2 * time.Second)
	}
}

// linked returns the bytes between two symbols of data_unitv.c.
func linked(start, end *C.uchar) []byte {
	p := unsafe.Pointer(start)
	return unsafe.Slice((*byte)(p), uintptr(unsafe.Pointer(end))-uintptr(p))
}

func run() error {
	model := linked(&C.ssd_model, &C.ssd_model_end)
	img := linked(&C.ssd_input, &C.ssd_input_end)
	in := unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(img))), len(img))
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	addr := uintptr(unsafe.Pointer(unsafe.SliceData(model)))
	println("ssd: boot model_addr", addr, "model_addr_mod128", int(addr%128), "model_bytes", len(model), "input_bytes", len(in),
		"heap_sys", ms.HeapSys, "heap_inuse", ms.HeapInuse)
	t0 := time.Now()
	f, err := kgraph.Parse(model)
	if err != nil {
		return err
	}
	println("ssd: parse ok ms", time.Since(t0).Milliseconds())
	d, err := newDetector(f, kpu.New)
	if err != nil {
		return err
	}
	return loop(os.Stdout, d, in, time.Since(t0), uint64(machine.CPUFrequency()), 0, mode == "dump")
}
