package main

import (
	"errors"
	"time"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kgraph"
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
	"github.com/itohio/EasyRobot/x/vision/detect/ssd"
)

// maxDetections caps the detections of a frame.
const maxDetections = 10

// Stages of a frame, for the time report.
const (
	stConvert  = iota // int8 NHWC ↔ uint8 CHW around every KPU segment
	stUpload          // CHW input into KPU memory
	stKPU             // the KPU's convolutions
	stReadback        // a segment's output out of KPU memory
	stCPUConv         // convolutions (and their Pads) on the CPU
	stTail            // reshapes, concatenations, logistic
	stDecode          // boxes and NMS
	nStages
)

var stageNames = [nStages]string{"convert", "upload", "kpu", "readback", "cpu_conv", "tail", "decode"}

// detector is SSD-MobileNetV1 run from its kgraph file: the KPU segments
// through one kpu.Runner each, the other layers on qnet's CPU kernels, the
// two outputs decoded with the anchors and constants of the file's extras.
type detector struct {
	file *kgraph.File
	be   *kgraph.Backend
	run  *qnet.Runner
	dec  *ssd.Decoder
	iou  float32

	runners []*kpu.Runner // one per KPU segment, in file order
	kpuSeg  []int         // per plan segment: its index in runners, or −1
	boxes   []nms.Box
	arena   int

	// Set by the caller: hash every segment's output into fnv; call each
	// with every segment's output right after it ran (a later segment may
	// reuse its memory).
	hash bool
	each func(seg int, out []int8)

	// Filled by frame.
	fnv    []uint32               // per plan segment (hash set)
	segT   []time.Duration        // per plan segment
	total  time.Duration          // the frame, without hashing and each
	ticks  [3]uint64              // upload, KPU, read-back in the runners' ticks
	inKPU  time.Duration          // wall time of the KPU segments
	stageT [nStages]time.Duration // see stages
}

var errExtras = errors.New("ssd: the model file has no anchors or decoder constants")

// newDetector prepares everything a frame needs from the parsed model file
// (whose bytes must stay alive); frame allocates nothing.
func newDetector(f *kgraph.File, newRunner kgraph.RunnerFunc) (*detector, error) {
	d := &detector{file: f}
	var err error
	d.be, err = kgraph.NewBackend(f, func(m *kmodel.Model, main []byte) (*kpu.Runner, error) {
		r, err := newRunner(m, main)
		if err == nil {
			d.runners = append(d.runners, r)
		}
		return r, err
	})
	if err != nil {
		return nil, err
	}
	g := &f.Graph
	p, err := qnet.PlanMemory(g, d.be)
	if err != nil {
		return nil, err
	}
	d.arena = p.Size()
	if d.run, err = qnet.NewRunner(g, make([]int8, d.arena), d.be); err != nil {
		return nil, err
	}
	anchors, ok := kgraph.ReadFloat32s(f.Extra("anchors"))
	k, ok2 := kgraph.ReadFloat32s(f.Extra("ssd"))
	if !ok || !ok2 || len(anchors) == 0 || len(k) != 5 || g.NumOutputs() != 2 {
		return nil, errExtras
	}
	bl, sl := &g.Layers[g.Outputs[0]], &g.Layers[g.Outputs[1]]
	d.dec, err = ssd.New(ssd.Config{Anchors: anchors, Classes: sl.Out.C - 1,
		Boxes:  ssd.Quant{Scale: bl.OutQ.Scale, ZP: int32(bl.OutQ.ZP)},
		Scores: ssd.Quant{Scale: sl.OutQ.Scale, ZP: int32(sl.OutQ.ZP)},
		YScale: k[0], XScale: k[1], HScale: k[2], WScale: k[3]})
	if err != nil {
		return nil, err
	}
	d.iou = k[4]
	segs := d.run.Plan().Segments
	d.kpuSeg, d.fnv, d.segT = make([]int, len(segs)), make([]uint32, len(segs)), make([]time.Duration, len(segs))
	n := 0
	for i, s := range segs {
		d.kpuSeg[i] = -1
		if s.Backend != nil {
			d.kpuSeg[i] = n
			n++
		}
	}
	d.boxes = make([]nms.Box, 0, maxDetections)
	return d, nil
}

// frame runs the graph on in (256×256×3 int8, the uint8 pixel − 128)
// segment by segment, timing each, and decodes the detections scoring at
// least threshold. The returned slice is reused by the next frame.
func (d *detector) frame(in []int8, threshold float32) ([]nms.Box, error) {
	g := &d.file.Graph
	d.ticks, d.inKPU, d.stageT = [3]uint64{}, 0, [nStages]time.Duration{}
	start := time.Now()
	var aside time.Duration // hashing and each
	for i, s := range d.run.Plan().Segments {
		t0 := time.Now()
		if err := d.run.RunSegment(in, i); err != nil {
			return nil, err
		}
		t1 := time.Now()
		dt := t1.Sub(t0)
		d.segT[i] = dt
		switch k := d.kpuSeg[i]; {
		case k >= 0:
			d.inKPU += dt
			for _, lt := range d.runners[k].Times() {
				d.ticks[0] += lt.Upload
				d.ticks[1] += lt.KPU
				d.ticks[2] += lt.Download
			}
		case g.Layers[s.First].Kind == qnet.Conv2D, g.Layers[s.First].Kind == qnet.DepthwiseConv2D, g.Layers[s.First].Kind == qnet.Pad:
			d.stageT[stCPUConv] += dt
		default:
			d.stageT[stTail] += dt
		}
		if d.hash || d.each != nil {
			out := d.run.SegmentOutput(in, i)
			if d.hash {
				d.fnv[i] = fnv1a(out)
			}
			if d.each != nil {
				d.each(i, out)
			}
			aside += time.Since(t1)
		}
	}
	t0 := time.Now()
	var err error
	d.boxes, err = d.dec.Decode(d.run.Output(in, 0), d.run.Output(in, 1), threshold, d.iou, d.boxes[:0])
	end := time.Now()
	d.stageT[stDecode] = end.Sub(t0)
	d.total = end.Sub(start) - aside
	return d.boxes, err
}

// split divides the KPU segments' wall time into upload, KPU and read-back
// from the runners' ticks (ticksPerSecond of them a second; 0 on a host,
// where the emulated KPU reports none), the rest being the layout
// conversions around each segment.
func (d *detector) split(ticksPerSecond uint64) {
	rest := d.inKPU
	if ticksPerSecond > 0 {
		for k, st := range [3]int{stUpload, stKPU, stReadback} {
			t := time.Duration(d.ticks[k] * uint64(time.Second/time.Microsecond) / ticksPerSecond * uint64(time.Microsecond))
			d.stageT[st] = t
			rest -= t
		}
	}
	d.stageT[stConvert] = rest
}

// fnv1a returns the 32-bit FNV-1a hash of b.
func fnv1a(b []int8) uint32 {
	h := uint32(2166136261)
	for _, v := range b {
		h = (h ^ uint32(uint8(v))) * 16777619
	}
	return h
}
