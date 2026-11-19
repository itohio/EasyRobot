package main

import (
	"io"
	"runtime"
	"strconv"
	"time"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// labels are the 80 COCO classes in the model's order.
var labels = [classes]string{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
	"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
	"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}

// reporter writes the result lines into one reused buffer, so a report
// allocates nothing and the heap figures show the detector alone.
type reporter struct {
	w   io.Writer
	buf []byte
	ms  runtime.MemStats
}

func (r *reporter) str(s string) *reporter { r.buf = append(r.buf, s...); return r }
func (r *reporter) i64(v int64) *reporter  { r.buf = strconv.AppendInt(r.buf, v, 10); return r }
func (r *reporter) u64(v uint64) *reporter { r.buf = strconv.AppendUint(r.buf, v, 10); return r }
func (r *reporter) hex(v uint32) *reporter {
	r.buf = strconv.AppendUint(r.buf, uint64(v), 16)
	return r
}
func (r *reporter) ms3(d time.Duration) *reporter {
	r.buf = strconv.AppendFloat(r.buf, float64(d.Microseconds())/1000, 'f', 3, 64)
	return r
}

func (r *reporter) flush() {
	r.buf = append(r.buf, '\r', '\n')
	r.w.Write(r.buf)
	r.buf = r.buf[:0]
}

// appendDet appends one detection in the host reference's format: anchor,
// label, score (4 decimals), box corners in input pixels (2 decimals).
func appendDet(b []byte, d nms.Box) []byte {
	b = strconv.AppendInt(b, int64(d.Index), 10)
	b = append(b, ' ')
	b = append(b, labels[d.Class]...)
	b = append(b, ' ')
	b = strconv.AppendFloat(b, float64(d.Score), 'f', 4, 32)
	for _, v := range [4]float32{d.X0, d.Y0, d.X1, d.Y1} {
		b = append(b, ' ')
		b = strconv.AppendFloat(b, float64(v), 'f', 2, 32)
	}
	return b
}

// header reports the setup: backend, model size and hash, input hash,
// arena, setup time (load, plan, prepare).
func (r *reporter) header(d *detector, in []int8, setup time.Duration) {
	r.str("yolo: backend ").str(qi8.Backend()).str(" layers ").i64(int64(len(d.g.Layers))).
		str(" segments ").i64(int64(len(d.seg))).str(" MACs ").i64(d.g.MACs()).
		str(" weights ").i64(int64(d.g.WeightBytes())).str(" model_fnv ").hex(d.modelFNV).
		str(" input_fnv ").hex(fnv1a(in)).str(" arena ").i64(int64(d.arena)).
		str(" setup_ms ").i64(setup.Milliseconds())
	r.flush()
}

// result reports run n: frame time (network + decode), decode time, output
// hashes, heap figures, detections and the network time per layer kind.
func (r *reporter) result(n int, d *detector, in []int8, dets []nms.Box, frame time.Duration) {
	runtime.ReadMemStats(&r.ms)
	r.str("yolo: run ").i64(int64(n)).str(" frame_ms ").ms3(frame).str(" decode_us ").i64(d.decode.Microseconds()).
		str(" fnv_scores ").hex(fnv1a(d.scores(in))).str(" fnv_boxes ").hex(fnv1a(d.boxes(in))).
		str(" dets ").i64(int64(len(dets))).
		str(" heap_inuse ").u64(r.ms.HeapInuse).str(" heap_sys ").u64(r.ms.HeapSys).
		str(" total_alloc ").u64(r.ms.TotalAlloc).str(" mallocs ").u64(r.ms.Mallocs)
	r.flush()
	for _, det := range dets {
		r.str("yolo: det ")
		r.buf = appendDet(r.buf, det)
		r.flush()
	}
	for b := range nBuckets {
		if d.layers[b] == 0 {
			continue
		}
		r.str("yolo: kind ").str(bucketName(b)).str(" layers ").i64(int64(d.layers[b])).
			str(" MACs ").i64(d.macs[b]).str(" ms ").ms3(d.elapsed[b])
		r.flush()
	}
}

// loop runs the detector on in and reports every run; runs ≤ 0 repeats
// forever (on the board, so a monitor can attach any time).
func loop(w io.Writer, d *detector, in []int8, setup time.Duration, runs int) error {
	r := &reporter{w: w, buf: make([]byte, 0, 256)}
	for n := 1; runs <= 0 || n <= runs; n++ {
		r.header(d, in, setup)
		t0 := time.Now()
		dets, err := d.detect(in)
		if err != nil {
			return err
		}
		r.result(n, d, in, dets, time.Since(t0))
	}
	return nil
}
