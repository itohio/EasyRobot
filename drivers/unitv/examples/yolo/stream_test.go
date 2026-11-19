package main

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

func testStream(t *testing.T) (*bytes.Buffer, []byte) {
	t.Helper()
	var out bytes.Buffer
	s := newStreamer(&out, frame.Meta{Width: 320, Height: 240, Format: frame.RGB565LE, Sensor: frame.SensorOV2640})
	img := fill565(320, 240, 10, 20, 30)
	lb := newLetterbox(320, 240)
	dets := []frame.Detection{
		lb.toImage(nms.Box{X0: 72.34, Y0: 17.21, X1: 399.54, Y1: 373.84, Score: 0.568, Class: 63}),
		lb.toImage(nms.Box{X0: 76.10, Y0: 28.83, X1: 385.05, Y1: 375.37, Score: 0.497, Class: 62}),
	}
	times := frame.StageTimes{Capture: 81_000, Letterbox: 412_000, Detect: 32_790_000, Send: 13_400_000}
	if err := s.send(img, 1234, dets, labels[:], times, [3]uint32{0xa1c21d60, 1, 2}); err != nil {
		t.Fatal(err)
	}
	return &out, img
}

// The stream carries the image, then its record, then log lines that
// state exactly what the record holds.
func TestStreamerImageRecordAndLog(t *testing.T) {
	out, img := testStream(t)
	var text bytes.Buffer
	dec := frame.NewDecoder(bytes.NewReader(out.Bytes()))
	dec.Text = &text
	im, err := dec.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(im.Data, img) || im.Meta.Width != 320 || im.TS != 1234 {
		t.Fatalf("image: %d bytes, %d px wide, ts %d", len(im.Data), im.Meta.Width, im.TS)
	}
	if _, err := dec.Next(); err != io.EOF {
		t.Fatalf("second Next: %v", err)
	}
	rec, n, err := frame.ParseDetections(text.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if rec.Seq != im.Seq || len(rec.Dets) != 2 || rec.Dets[0].Class != 63 || rec.Times.Detect != 32_790_000 {
		t.Fatalf("record %+v", rec)
	}
	log := string(text.Bytes()[n:])
	want := string(appendRecordLog(nil, &rec, labels[:]))
	if !strings.HasPrefix(log, want) {
		t.Errorf("log after the record:\n%q\ndoes not start with the decoded record's lines:\n%q", log, want)
	}
	if !strings.Contains(log, "yolo: det 0 laptop 0.5680 56 13 307 288\r\n") {
		t.Errorf("log lacks the laptop line in image pixels:\n%s", log)
	}
	if !strings.Contains(log, "yolo: check input_fnv a1c21d60 out0_fnv 1 out1_fnv 2 total_alloc ") {
		t.Errorf("log lacks the hash line:\n%s", log)
	}
}

// Class names come from the detector's table; ids outside it print as
// "class<id>", and at most frame.MaxDetections detections are sent.
func TestStreamerUsesDetectorLabels(t *testing.T) {
	var out bytes.Buffer
	s := newStreamer(&out, frame.Meta{Width: 4, Height: 2, Format: frame.Y8})
	dets := make([]frame.Detection, frame.MaxDetections+3)
	dets[0].Class, dets[1].Class = 1, 5
	if err := s.send(make([]byte, 8), 0, dets, []string{"zero", "one"}, frame.StageTimes{}, [3]uint32{}); err != nil {
		t.Fatal(err)
	}
	log := out.String()
	if !strings.Contains(log, "yolo: det 0 one 0.0000 0 0 0 0\r\n") || !strings.Contains(log, "yolo: det 0 class5 ") {
		t.Errorf("log:\n%s", log)
	}
	if !strings.Contains(log, "yolo: frame 0 dets 64 ") {
		t.Errorf("frame line does not cap the count at %d:\n%s", frame.MaxDetections, log)
	}
}

func TestStreamerSendAllocatesNothing(t *testing.T) {
	s := newStreamer(io.Discard, frame.Meta{Width: 320, Height: 240, Format: frame.RGB565LE})
	img := fill565(320, 240, 1, 2, 3)
	dets := []frame.Detection{{X1: 10, Y1: 10, Score: 60000}}
	if n := testing.AllocsPerRun(5, func() { s.send(img, 0, dets, labels[:], frame.StageTimes{}, [3]uint32{}) }); n != 0 {
		t.Errorf("%v allocations per send", n)
	}
}
