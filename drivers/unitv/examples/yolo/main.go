//go:build unitv

// Command yolo runs YOLOX-nano (int8, 416×416, COCO) on the UnitV's CPU
// with qnet's kernels, on UARTHS (USB serial, 115200 baud). The activation
// arena is the KPU's 2 MiB memory, lent to the CPU (kpu.LendMemory): the
// heap has room for the graph but not for the arena in one piece. The model
// (and the test image) are the constants of data_unitv.go, written by gen.go
// (README.md).
//
// Build-time option source (tinygo -ldflags "-X main.source=image"):
//   - camera (default): capture a QVGA RGB565 frame, letterbox it, detect,
//     and write the frame as a frame v1 image followed by its detection
//     record (drivers/camera/frame) and log lines, for web/index.html;
//   - image: run on the built-in test image and print, every run, the
//     frame time, FNV-1a hashes of both output tensors, heap figures, the
//     detections and the time per layer kind. The host build (host.go)
//     prints the same lines, the reference the board must reproduce.
package main

import (
	"machine"
	"os"
	"runtime"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/unitv"
)

var source = "camera"

func main() {
	clearStack()
	var err error
	if source == "image" {
		err = runImage()
	} else {
		err = runCamera()
	}
	for err != nil {
		println("yolo: error", err.Error())
		time.Sleep(2 * time.Second)
	}
}

// detector runs in the lent KPU memory; the loader aliases the int8
// weights of the model constant, the only copy.
func setupDetector() (*detector, error) {
	mem, err := kpu.LendMemory()
	if err != nil {
		return nil, err
	}
	arena := unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(mem))), len(mem))
	return newDetector(unsafe.Slice(unsafe.StringData(model), len(model)), arena)
}

func runImage() error {
	t0 := time.Now()
	d, err := setupDetector()
	if err != nil {
		return err
	}
	setup := time.Since(t0)
	// Run only reads its input: the constant is used in place.
	in := unsafe.Slice((*int8)(unsafe.Pointer(unsafe.StringData(input))), len(input))
	return loop(os.Stdout, d, in, setup, 0)
}

// align8 returns an 8-byte aligned window of n bytes (the DVP writes with
// 64-bit AXI bursts).
func align8(n int) []byte {
	b := make([]byte, n+8)
	off := int(-uintptr(unsafe.Pointer(&b[0])) & 7)
	return b[off : off+n]
}

// runCamera: QVGA RGB565, because the detector needs colour and the page
// shows the image the detector saw; 153,600 bytes take ≈ 13.3 s at
// 115200 baud, which is loss-free without chunk resend and short beside a
// ≈ 33 s detection.
func runCamera() error {
	uart := machine.UART0
	uart.Configure(machine.UARTConfig{BaudRate: 115200})
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	heapSys, freeAtStart := ms.HeapSys, ms.HeapIdle
	cfg := camera.Config{Mode: sensor.Mode{Resolution: sensor.QVGA, Format: sensor.RGB565}}
	cam, err := camera.Open(cfg, unitv.CameraPins)
	if err != nil {
		return err
	}
	// The big buffers first, on an empty heap, before the loader's garbage.
	buf := align8(camera.FrameSize(cfg))
	in := make([]int8, inputSize*inputSize*3)
	t0 := time.Now()
	d, err := setupDetector()
	if err != nil {
		return err
	}
	var det Detector = newYOLOXObjects(d, in, int(cfg.W), int(cfg.H))
	setup := time.Since(t0)
	runtime.GC()
	runtime.ReadMemStats(&ms)
	println("yolo: camera", cam.Sensor().Model.String(), "qvga rgb565, setup_ms", setup.Milliseconds(),
		"heap_sys", heapSys, "free_at_start", freeAtStart, "live_after_setup", ms.HeapInuse)
	s := newStreamer(uart, frame.Meta{Width: cfg.W, Height: cfg.H, Format: frame.RGB565LE, Sensor: frame.Sensor(cam.Sensor().Model)})
	for range 10 { // let auto exposure settle
		cam.Capture(buf)
	}
	start := time.Now()
	var sendPrev time.Duration
	for {
		t0 := time.Now()
		fr, err := cam.Capture(buf)
		if err != nil {
			println("yolo: capture:", err.Error())
			continue
		}
		t1 := time.Now()
		dets, err := det.Detect(fr.Data)
		if err != nil {
			return err
		}
		prep, infer := det.Stages()
		times := frame.StageTimes{Capture: us(t1.Sub(t0)), Letterbox: us(prep), Detect: us(infer), Send: us(sendPrev)}
		fnv := det.Check()
		t2 := time.Now()
		if err := s.send(fr.Data, us(t0.Sub(start)), dets, det.Labels(), times, fnv); err != nil {
			println("yolo: send:", err.Error())
		}
		sendPrev = time.Since(t2)
	}
}

func us(d time.Duration) uint32 { return uint32(d.Microseconds()) }
