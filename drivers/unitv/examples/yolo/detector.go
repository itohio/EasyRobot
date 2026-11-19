package main

import (
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/nn/qnet/tfl"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
	"github.com/itohio/EasyRobot/x/vision/detect/yolox"
)

// Model geometry and decode thresholds (YOLOX-nano, COCO, 416×416).
const (
	inputSize = 416
	classes   = 80
	confMin   = 0.3
	iouMax    = 0.45
	maxCand   = 300 // decoder candidates above confMin kept per frame
	maxDets   = 50
)

// pointwise is the timing bucket of 1×1 Conv2D layers; the other buckets are
// the qnet.Kind values themselves.
const pointwise = 0

// nBuckets covers every qnet.Kind value (a uint8; valid kinds start at 1)
// plus pointwise, so kinds added to qnet later get their own bucket.
const nBuckets = 1 << 8

// bucket returns the timing bucket of l: its qnet.Kind, except that a 1×1
// Conv2D counts as pointwise.
func bucket(l *qnet.Layer) int {
	if l.Kind == qnet.Conv2D && l.KH == 1 && l.KW == 1 {
		return pointwise
	}
	return int(l.Kind)
}

// bucketName names a bucket.
func bucketName(b int) string {
	if b == pointwise {
		return "pointwise"
	}
	return qnet.Kind(b).String()
}

// detector runs YOLOX-nano on qnet's CPU kernels: the model is loaded once,
// the arena and every buffer are allocated in newDetector, and detect
// allocates nothing.
type detector struct {
	g      qnet.Graph
	run    *qnet.Runner
	dec    *yolox.Decoder
	si, bi int     // graph outputs: scores [anchors×80], boxes [anchors×4]
	seg    []uint8 // timing bucket of each plan segment
	dets   []nms.Box

	modelFNV uint32                  // FNV-1a of the model bytes
	arena    int                     // bytes
	layers   [nBuckets]int           // layers per bucket
	macs     [nBuckets]int64         // nominal MACs per bucket
	elapsed  [nBuckets]time.Duration // time per bucket in the last detect
	decode   time.Duration           // decode + NMS time in the last detect
}

// newDetector loads the TFLite model (its int8 weights alias model, which
// must stay unchanged while the detector is used) and allocates the arena.
func newDetector(model []byte) (*detector, error) {
	d := &detector{modelFNV: fnv1a(unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(model))), len(model)))}
	var err error
	if d.g, err = tfl.Load(model); err != nil {
		return nil, err
	}
	plan, err := qnet.PlanMemory(&d.g)
	if err != nil {
		return nil, err
	}
	d.arena = plan.Size()
	if d.run, err = qnet.NewRunner(&d.g, make([]int8, d.arena)); err != nil {
		return nil, err
	}
	d.si, d.bi = 0, 1
	if d.g.Layers[d.g.Outputs[0]].Out.C != classes {
		d.si, d.bi = 1, 0
	}
	sq, bq := d.g.Layers[d.g.Outputs[d.si]].OutQ, d.g.Layers[d.g.Outputs[d.bi]].OutQ
	if d.dec, err = yolox.New(yolox.Config{W: inputSize, H: inputSize, Strides: []int{8, 16, 32}, Classes: classes,
		Scores: yolox.Quant{Scale: sq.Scale, ZP: int32(sq.ZP)}, Boxes: yolox.Quant{Scale: bq.Scale, ZP: int32(bq.ZP)}}, maxCand); err != nil {
		return nil, err
	}
	for _, s := range d.run.Plan().Segments {
		d.seg = append(d.seg, uint8(bucket(&d.g.Layers[s.First])))
	}
	for i := range d.g.Layers {
		b := bucket(&d.g.Layers[i])
		d.layers[b]++
		d.macs[b] += d.g.Budget(i).MACs
	}
	d.dets = make([]nms.Box, 0, maxDets)
	return d, nil
}

// detect runs the network on in (416×416×3 int8: uint8 RGB − 128), timing each
// plan segment into d.elapsed, and decodes the detections (score ≥ confMin,
// NMS at iouMax) timed into d.decode. The result aliases d and is valid
// until the next detect.
func (d *detector) detect(in []int8) ([]nms.Box, error) {
	d.elapsed = [nBuckets]time.Duration{}
	for k, b := range d.seg {
		t0 := time.Now()
		if err := d.run.RunSegment(in, k); err != nil {
			return nil, err
		}
		d.elapsed[b] += time.Since(t0)
	}
	t0 := time.Now()
	dets, err := d.dec.Decode(d.scores(in), d.boxes(in), confMin, iouMax, d.dets[:0])
	d.decode = time.Since(t0)
	return dets, err
}

// scores and boxes return the raw output tensors of the last detect.
func (d *detector) scores(in []int8) []int8 { return d.run.Output(in, d.si) }
func (d *detector) boxes(in []int8) []int8  { return d.run.Output(in, d.bi) }

// fnv1a returns the 32-bit FNV-1a hash of b.
func fnv1a(b []int8) uint32 {
	h := uint32(2166136261)
	for _, v := range b {
		h = (h ^ uint32(uint8(v))) * 16777619
	}
	return h
}
