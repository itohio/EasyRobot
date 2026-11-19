package tfl

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
	"github.com/itohio/EasyRobot/x/vision/detect/ssd"
)

// TestSSDTailMatchesInterpreter loads testdata/ssdtail.tflite
// (gen_ssdtail.py: the tail of a quantised SSD-MobileNetV1 export — uint8
// box and class heads, CONCATENATIONs of differently quantised inputs,
// LOGISTIC, then the custom TFLite_Detection_PostProcess) with
// LoadDetection. Every layer output of the qnet Runner must equal the LiteRT
// reference interpreter's operator output, and ssd.Decoder on the two graph
// outputs must return the custom operator's detections: same count, classes
// and scores, boxes equal to float32 rounding (ssdtail.json.gz, 64 inputs).
func TestSSDTailMatchesInterpreter(t *testing.T) {
	model, err := os.ReadFile("testdata/ssdtail.tflite")
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Ops   []string `json:"ops"`
		Cases []struct {
			Input string   `json:"input"`
			Outs  []string `json:"outs"`
			Post  []string `json:"post"`
		} `json:"cases"`
	}
	readGz(t, "testdata/ssdtail.json.gz", &fx)
	if _, err := Load(model); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Load: %v, want ErrUnsupported (the custom operator is cut only by LoadDetection)", err)
	}
	g, pp, err := LoadDetection(model)
	if err != nil {
		t.Fatal(err)
	}
	want := []qnet.Kind{qnet.Conv2D, qnet.Conv2D, qnet.Reshape, qnet.Conv2D, qnet.Reshape, qnet.Conv2D, qnet.Conv2D, qnet.Reshape,
		qnet.Conv2D, qnet.Reshape, qnet.Concat, qnet.Concat, qnet.Reshape, qnet.Logistic}
	if len(g.Layers) != len(want) || len(fx.Ops) != len(want) {
		t.Fatalf("%d layers, %d operators, want %d", len(g.Layers), len(fx.Ops), len(want))
	}
	for i := range want {
		if g.Layers[i].Kind != want[i] {
			t.Fatalf("layer %d is %v (operator %s), want %v", i, g.Layers[i].Kind, fx.Ops[i], want[i])
		}
	}
	if len(g.Outputs) != 2 || g.Outputs[0] != 12 || g.Outputs[1] != 13 {
		t.Fatalf("outputs %v, want [12 13] (box encodings, class scores)", g.Outputs)
	}
	wantPP := PostProcess{MaxDetections: 5, MaxClassesPerDetection: 1, DetectionsPerClass: 100, NumClasses: 2,
		ScoreThreshold: 179.0 / 256, IoUThreshold: 0.5, YScale: 10, XScale: 10, HScale: 5, WScale: 5}
	got := pp
	got.Anchors = nil
	if !reflect.DeepEqual(got, wantPP) {
		t.Fatalf("post-process options %+v, want %+v", got, wantPP)
	}
	if len(pp.Anchors) != 40*4 || pp.Anchors[0] != 16.0/128 || pp.Anchors[3] != 26.0/128 {
		t.Fatalf("anchors: %d values, first row %v, want 160 starting [0.125 0.125 0.25 0.203125]", len(pp.Anchors), pp.Anchors[:min(4, len(pp.Anchors))])
	}
	bq, sq := g.Layers[12].OutQ, g.Layers[13].OutQ
	dec, err := ssd.New(ssd.Config{
		Anchors: pp.Anchors, Classes: pp.NumClasses,
		Boxes:  ssd.Quant{Scale: bq.Scale, ZP: int32(bq.ZP)},
		Scores: ssd.Quant{Scale: sq.Scale, ZP: int32(sq.ZP)},
		YScale: pp.YScale, XScale: pp.XScale, HScale: pp.HScale, WScale: pp.WScale,
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := qnet.NewRunner(&g, make([]int8, 1<<14))
	if err != nil {
		t.Fatal(err)
	}
	dec64 := func(s string) []byte {
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	out := make([]nms.Box, 0, pp.MaxDetections)
	for ci, c := range fx.Cases {
		in := i8(dec64(c.Input))
		for s, seg := range r.Plan().Segments {
			if err := r.RunSegment(in, s); err != nil {
				t.Fatal(err)
			}
			i := seg.First + seg.Count - 1
			if got, exp := r.SegmentOutput(in, s), i8(dec64(c.Outs[i])); !bytes.Equal(u8(got), u8(exp)) {
				t.Fatalf("case %d layer %d (%s): %d bytes differ from the interpreter", ci, i, fx.Ops[i], diffCount(got, exp))
			}
		}
		if _, err := r.Run(in); err != nil {
			t.Fatal(err)
		}
		dets, err := dec.Decode(r.Output(in, 0), r.Output(in, 1), pp.ScoreThreshold, pp.IoUThreshold, out[:0])
		if err != nil {
			t.Fatal(err)
		}
		boxes, classes, scores, num := f32s(dec64(c.Post[0])), f32s(dec64(c.Post[1])), f32s(dec64(c.Post[2])), f32s(dec64(c.Post[3]))
		if len(dets) != int(num[0]) {
			t.Fatalf("case %d: %d detections, want %v", ci, len(dets), num[0])
		}
		for k, d := range dets {
			b := boxes[4*k : 4*k+4]
			if float32(d.Class) != classes[k] || d.Score != scores[k] ||
				!near(d.Y0, b[0]) || !near(d.X0, b[1]) || !near(d.Y1, b[2]) || !near(d.X1, b[3]) {
				t.Fatalf("case %d detection %d: %+v, want class %v score %v box (y0 x0 y1 x1) %v", ci, k, d, classes[k], scores[k], b)
			}
		}
	}
}

// near: equal within one float32 rounding step.
func near(a, b float32) bool {
	return a == b || math.Nextafter32(a, b) == b
}

func f32s(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func readGz(t *testing.T, path string, v any) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}
