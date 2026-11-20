//go:build !unitv

package main

import (
	"math"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/compile"
	"github.com/itohio/EasyRobot/drivers/kpu/kgraph"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/nn/qnet/tfl"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
	"github.com/itohio/EasyRobot/x/vision/detect/ssd"
)

// miniSSD builds a detector shaped as SSD-MobileNetV1's tail at 10×10: two
// feature maps (10×10 and 5×5), on each a box head and a score head with
// the generator's box counts (3 a cell on the first map, 6 on the second,
// 2 classes plus background), reshapes, concatenations and a logistic; and
// its post-process, with the anchors TF's generator gives those maps.
func miniSSD(t *testing.T) (*qnet.Graph, tfl.PostProcess) {
	t.Helper()
	const c, n = 4, 10
	q := qnet.Quant{Scale: 0.05, ZP: -3}
	conv := func(from []int, in qi8.Shape, outC, k, stride int) qnet.Layer {
		l := qnet.Layer{Kind: qnet.Conv2D, From: from, In: in, Out: qi8.Shape{C: outC}, InQ: q, OutQ: q,
			KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
		out, err := l.OutShape()
		if err != nil {
			t.Fatal(err)
		}
		l.Out = out
		return l
	}
	reshape := func(in, out qi8.Shape) qnet.Layer {
		return qnet.Layer{Kind: qnet.Reshape, In: in, Out: out, InQ: q, OutQ: q}
	}
	f1 := conv(nil, qi8.Shape{H: n, W: n, C: 3}, c, 3, 1) // 0
	f2 := conv(nil, f1.Out, c, 3, 2)                      // 1
	n1, n2 := f1.Out.H*f1.Out.W*3, f2.Out.H*f2.Out.W*6
	b1 := conv([]int{0}, f1.Out, 3*4, 1, 1) // 2
	b2 := conv([]int{1}, f2.Out, 6*4, 1, 1) // 4
	c1 := conv([]int{0}, f1.Out, 3*3, 1, 1) // 6
	c2 := conv([]int{1}, f2.Out, 6*3, 1, 1) // 8
	lp, err := qi8.QuantizeLogistic(q.Scale, q.ZP)
	if err != nil {
		t.Fatal(err)
	}
	boxes, scores := qi8.Shape{H: n1 + n2, W: 1, C: 4}, qi8.Shape{H: 1, W: n1 + n2, C: 3}
	g := &qnet.Graph{Layers: []qnet.Layer{
		f1, f2,
		b1, reshape(b1.Out, qi8.Shape{H: n1, W: 1, C: 4}),
		b2, reshape(b2.Out, qi8.Shape{H: n2, W: 1, C: 4}),
		c1, reshape(c1.Out, qi8.Shape{H: 1, W: n1, C: 3}),
		c2, reshape(c2.Out, qi8.Shape{H: 1, W: n2, C: 3}),
		{Kind: qnet.Concat, From: []int{3, 5}, Axis: 0, In: qi8.Shape{H: n1, W: 1, C: 4}, Out: boxes, InQ: q, OutQ: q},
		reshape(boxes, qi8.Shape{H: 1, W: n1 + n2, C: 4}),
		{Kind: qnet.Concat, From: []int{7, 9}, Axis: 1, In: qi8.Shape{H: 1, W: n1, C: 3}, Out: scores, InQ: q, OutQ: q},
		{Kind: qnet.Logistic, In: scores, Out: scores, InQ: q, OutQ: qnet.LogisticOutQuant, LogisticParams: &lp},
	}, Outputs: []int{11, 13}}
	for i := range g.Layers {
		l := &g.Layers[i]
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	}
	if err := qnet.SyntheticParams(g, 7); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	anchors, err := ssd.Anchors(ssd.MobileNetV1Anchors([][2]int{{10, 10}, {5, 5}}))
	if err != nil {
		t.Fatal(err)
	}
	return g, tfl.PostProcess{Anchors: anchors, NumClasses: 2, YScale: 10, XScale: 10, HScale: 5, WScale: 5, IoUThreshold: 0.6}
}

// TestBuildModel: the file written for an 8×8 input loads without the
// compiler and gives the outputs of the graph compiled directly; its
// anchors are the trained ones of the cells that remain (8×8 of 10×10 and
// 4×4 of 5×5), and its constants the post-process's.
func TestBuildModel(t *testing.T) {
	g, pp := miniSSD(t)
	data, err := buildModel(g, pp, 8)
	if err != nil {
		t.Fatal(err)
	}
	f, err := kgraph.Parse(kpu.AlignedCopy(data))
	if err != nil {
		t.Fatal(err)
	}
	d, err := newDetector(f, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ssd.CropAnchors(ssd.MobileNetV1Anchors([][2]int{{10, 10}, {5, 5}}), [2]int{10, 10}, [2]int{8, 8}, [][2]int{{8, 8}, {4, 4}})
	if err != nil {
		t.Fatal(err)
	}
	if d.dec.Anchors() != len(want)/4 || len(want)/4 != 8*8*3+4*4*6 {
		t.Fatalf("%d anchors, want %d", d.dec.Anchors(), len(want)/4)
	}
	got, _ := kgraph.ReadFloat32s(d.file.Extra("anchors"))
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("anchor value %d: %v, want %v", i, got[i], want[i])
		}
	}
	if d.iou != pp.IoUThreshold {
		t.Errorf("IoU threshold %v, want %v", d.iou, pp.IoUThreshold)
	}

	r, err := compile.ResizeAligned(g, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	be, err := compile.Compile(r, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	p, err := qnet.PlanMemory(r, be)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := qnet.NewRunner(r, make([]int8, p.Size()), be)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]int8, 8*8*3)
	for i := range in {
		in[i] = int8(i*53 + i/7)
	}
	if _, err := ref.Run(in); err != nil {
		t.Fatal(err)
	}
	d.hash = true
	dets, err := d.frame(in, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	for k := range 2 {
		a, b := d.run.Output(in, k), ref.Output(in, k)
		if len(a) != len(b) {
			t.Fatalf("output %d: %d values, want %d", k, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("output %d value %d: %d, the compiled graph gives %d", k, i, a[i], b[i])
			}
		}
	}
	// The decoder is the one the file describes: the same boxes as a
	// decoder built from the graph and the anchors directly.
	bl, sl := &r.Layers[r.Outputs[0]], &r.Layers[r.Outputs[1]]
	dec, err := ssd.New(ssd.Config{Anchors: want, Classes: 2,
		Boxes: ssd.Quant{Scale: bl.OutQ.Scale, ZP: int32(bl.OutQ.ZP)}, Scores: ssd.Quant{Scale: sl.OutQ.Scale, ZP: int32(sl.OutQ.ZP)},
		YScale: 10, XScale: 10, HScale: 5, WScale: 5})
	if err != nil {
		t.Fatal(err)
	}
	wantDets, err := dec.Decode(ref.Output(in, 0), ref.Output(in, 1), 0.01, pp.IoUThreshold, make([]nms.Box, 0, maxDetections))
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) == 0 || len(dets) != min(len(wantDets), maxDetections) {
		t.Fatalf("%d detections, want %d (at most %d)", len(dets), len(wantDets), maxDetections)
	}
	for i := range dets {
		if dets[i] != wantDets[i] {
			t.Errorf("detection %d: %+v, want %+v", i, dets[i], wantDets[i])
		}
	}
	// Every segment was hashed, and the last one's hash is the scores'.
	last := len(d.fnv) - 1
	if d.fnv[last] != fnv1a(ref.Output(in, 1)) || d.fnv[0] == 0 {
		t.Errorf("segment hashes %x", d.fnv)
	}
}

// TestBuildModelRefuses: anchors that are not the generator's for the head
// maps (another model family), and a graph without the two head outputs.
func TestBuildModelRefuses(t *testing.T) {
	g, pp := miniSSD(t)
	moved := pp
	moved.Anchors = append([]float32(nil), pp.Anchors...)
	moved.Anchors[5] += 0.02
	if _, err := buildModel(g, moved, 8); err == nil || !strings.Contains(err.Error(), "anchors") {
		t.Errorf("anchors moved by 0.02: %v", err)
	}
	short := pp
	short.Anchors = pp.Anchors[:len(pp.Anchors)-4]
	if _, err := buildModel(g, short, 8); err == nil || !strings.Contains(err.Error(), "anchors") {
		t.Errorf("one anchor short: %v", err)
	}
	one := *g
	one.Outputs = []int{13}
	if _, err := buildModel(&one, pp, 8); err == nil {
		t.Error("a graph with one output was accepted")
	}
	// The stored anchors are uint8: half a step of rounding is accepted.
	rounded := pp
	rounded.Anchors = make([]float32, len(pp.Anchors))
	for i, v := range pp.Anchors {
		rounded.Anchors[i] = float32(math.Round(float64(v)/0.0064530605) * 0.0064530605)
	}
	if _, err := buildModel(g, rounded, 8); err != nil {
		t.Errorf("anchors rounded to the export's uint8 steps: %v", err)
	}
}

func TestFNV1a(t *testing.T) {
	// FNV-1a test vectors: "" and "a".
	if got := fnv1a(nil); got != 0x811c9dc5 {
		t.Errorf("empty: %x", got)
	}
	if got := fnv1a([]int8{'a'}); got != 0xe40c292c {
		t.Errorf("a: %x", got)
	}
}
