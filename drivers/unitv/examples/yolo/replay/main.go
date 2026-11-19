// Command replay writes web/replay.bin, the stream the viewer page
// (web/index.html, "Open replay") can show without a board: three frames
// of a synthetic 160×120 RGB565 scene with made-up detections, in the
// format the yolo firmware writes (frame v1 images, each followed by its
// detection record and log lines). Frame 1 has a corrupted chunk, so the
// page shows how a dropped frame looks; its record then belongs to no image.
//
//	go run ./drivers/unitv/examples/yolo/replay
package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

const w, h = 160, 120

// records are the detection records of frames 0, 1 and 2 (class ids are
// COCO: 2 car, 56 chair, 58 potted plant).
var records = []frame.Detections{
	{Seq: 0, Width: w, Height: h, Times: frame.StageTimes{Capture: 21_000, Letterbox: 98_000, Detect: 32_788_000},
		Dets: []frame.Detection{{Class: 2, Score: 59637, X0: 18, Y0: 62, X1: 78, Y1: 98}, {Class: 56, Score: 27525, X0: 104, Y0: 40, X1: 140, Y1: 100}}},
	{Seq: 1, Width: w, Height: h, Times: frame.StageTimes{Capture: 21_000, Letterbox: 98_000, Detect: 32_790_000, Send: 3_410_000},
		Dets: []frame.Detection{{Class: 2, Score: 58982, X0: 22, Y0: 62, X1: 82, Y1: 98}}},
	{Seq: 2, Width: w, Height: h, Times: frame.StageTimes{Capture: 21_000, Letterbox: 98_000, Detect: 32_787_000, Send: 3_410_000},
		Dets: []frame.Detection{{Class: 58, Score: 45875, X0: 100, Y0: 30, X1: 150, Y1: 112}}},
}

// corruptFrame is the frame whose second chunk gets a flipped bit.
const corruptFrame = 1

func main() {
	out := filepath.Join("drivers", "unitv", "examples", "yolo", "web", "replay.bin")
	if len(os.Args) > 1 {
		out = os.Args[1]
	}
	if err := os.WriteFile(out, build(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "replay:", err)
		os.Exit(1)
	}
}

// build returns the replay stream; it is deterministic.
func build() []byte {
	var s, img bytes.Buffer
	s.WriteString("yolo: replay file: synthetic scene, made-up detections (replay/main.go)\r\n")
	enc := frame.NewEncoder(&img) // numbers the images 0, 1, 2 like records
	for i, r := range records {
		img.Reset()
		if err := enc.WriteImage(frame.Meta{Width: w, Height: h, Format: frame.RGB565LE, Sensor: frame.SensorOV2640},
			uint32(i)*36_400_000, scene(i)); err != nil {
			panic(err)
		}
		b := img.Bytes()
		if i == corruptFrame {
			b[frame.HeaderSize+frame.MaxPayload+frame.HeaderSize+100] ^= 0x01 // in chunk 1's payload
		}
		s.Write(b)
		s.Write(frame.AppendDetections(nil, &r))
		fmt.Fprintf(&s, "yolo: frame %d dets %d detect_ms %.1f\r\n", r.Seq, len(r.Dets), float64(r.Times.Detect)/1000)
	}
	return s.Bytes()
}

// scene draws frame i: a sky-to-ground gradient, a red block (the "car")
// that moves 4 px per frame, a brown frame (the "chair") and, from frame 2,
// a green blob (the "plant").
func scene(i int) []byte {
	img := make([]byte, 2*w*h)
	set := func(x, y int, r, g, b uint8) {
		p := uint16(r>>3)<<11 | uint16(g>>2)<<5 | uint16(b>>3)
		img[2*(y*w+x)], img[2*(y*w+x)+1] = byte(p), byte(p>>8)
	}
	for y := range h {
		for x := range w {
			if y < h/2 {
				set(x, y, uint8(80+y), uint8(140+y), 230)
			} else {
				set(x, y, 90, uint8(120-(y-h/2)), 60)
			}
		}
	}
	rect := func(x0, y0, x1, y1 int, r, g, b uint8) {
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				set(x, y, r, g, b)
			}
		}
	}
	dx := 4 * i
	rect(20+dx, 70, 76+dx, 92, 200, 30, 30)
	rect(26+dx, 64, 66+dx, 72, 170, 20, 20)
	rect(24+dx, 90, 34+dx, 98, 20, 20, 20)
	rect(62+dx, 90, 72+dx, 98, 20, 20, 20)
	if i < 2 {
		rect(106, 42, 110, 100, 120, 70, 30)
		rect(134, 42, 138, 100, 120, 70, 30)
		rect(106, 70, 138, 76, 120, 70, 30)
	} else {
		rect(116, 80, 134, 110, 150, 90, 50)
		for y := 32; y < 82; y++ {
			for x := 102; x < 148; x++ {
				if dx, dy := x-125, y-56; dx*dx+dy*dy*2 < 22*22 {
					set(x, y, 40, uint8(150+(x+y)%40), 50)
				}
			}
		}
	}
	return img
}
