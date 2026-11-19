// Package ssd decodes the two head outputs of a quantised SSD detector
// (TF Object Detection API exports such as SSD-MobileNetV1 COCO) into boxes,
// as TensorFlow Lite's TFLite_Detection_PostProcess operator does on its
// fast path (use_regular_nms false, one class a box): box encodings
// [anchors × 4] (ty, tx, th, tw, centre-size coded against the anchors) and
// class scores [anchors × (1 + classes)] (column 0 is the background).
// qnet/tfl's LoadDetection cuts that operator off a model and returns its
// anchors and options.
//
// For each anchor the best class (the first of equal scores) is a candidate
// when its score reaches the threshold; candidates go through greedy NMS
// (nms.Greedy, ties by lower anchor index) regardless of class. Scores equal
// TFLite's exactly; boxes are computed in float64 as TFLite does and can
// differ in the last float32 bit where math.Exp and the C library's exp
// round differently. After New, Decode allocates nothing.
package ssd

import (
	"errors"
	"math"

	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// Quant is a tensor's int8 quantisation: real = Scale · (q − ZP). TFLite's
// uint8 tensors, loaded as int8 (value − 128), have ZP = uint8 ZP − 128.
type Quant struct {
	Scale float32
	ZP    int32
}

// Config describes the model.
type Config struct {
	// Anchors holds one [ycenter, xcenter, h, w] row per box (normalised).
	Anchors []float32
	// Classes is the number of classes, background excluded.
	Classes int
	Boxes   Quant // box encodings tensor
	Scores  Quant // class scores tensor (after the logistic)
	// YScale, XScale, HScale, WScale divide the encodings (10, 10, 5, 5 in
	// the TF exports).
	YScale, XScale, HScale, WScale float32
}

// Decoder decodes one model's outputs. It is not safe for concurrent use.
type Decoder struct {
	cfg   Config
	n     int
	score [256]float32 // Scores.Scale·(q−ZP), indexed by q+128
	box   [256]float32 // Boxes.Scale·(q−ZP)
	cand  []nms.Box
}

// ErrConfig reports a configuration that cannot describe an SSD head.
var ErrConfig = errors.New("ssd: invalid config (anchors a non-empty multiple of 4, classes > 0, positive scales)")

// ErrLength reports output tensors whose lengths do not match the config.
var ErrLength = errors.New("ssd: box or score tensor length does not match the anchors and classes")

// New returns a Decoder for cfg.
func New(cfg Config) (*Decoder, error) {
	ok := func(v float32) bool { return v > 0 && !math.IsInf(float64(v), 0) }
	if len(cfg.Anchors) == 0 || len(cfg.Anchors)%4 != 0 || cfg.Classes <= 0 ||
		!ok(cfg.Boxes.Scale) || !ok(cfg.Scores.Scale) || !ok(cfg.YScale) || !ok(cfg.XScale) || !ok(cfg.HScale) || !ok(cfg.WScale) {
		return nil, ErrConfig
	}
	n := len(cfg.Anchors) / 4
	d := &Decoder{cfg: cfg, n: n, cand: make([]nms.Box, 0, n)}
	for i := range 256 {
		q := float32(int32(i-128) - cfg.Scores.ZP)
		d.score[i] = q * cfg.Scores.Scale
		q = float32(int32(i-128) - cfg.Boxes.ZP)
		d.box[i] = q * cfg.Boxes.Scale
	}
	return d, nil
}

// Anchors is the number of boxes the model predicts.
func (d *Decoder) Anchors() int { return d.n }

// Decode appends to out (up to cap(out), the best first) the detections in
// boxes (anchors × 4) and scores (anchors × (1 + Classes)) whose best class
// score is at least threshold, after NMS at iou. Box corners are normalised
// (X0, Y0 top left); Class counts from 0 without the background; Index is
// the anchor.
func (d *Decoder) Decode(boxes, scores []int8, threshold, iou float32, out []nms.Box) ([]nms.Box, error) {
	c := d.cfg.Classes + 1
	if len(boxes) != 4*d.n || len(scores) != c*d.n {
		return out, ErrLength
	}
	cand := d.cand[:0]
	for a := range d.n {
		row := scores[a*c+1 : a*c+c]
		best := 0
		for k := 1; k < len(row); k++ {
			if row[k] > row[best] {
				best = k
			}
		}
		s := d.score[int(row[best])+128]
		if s < threshold {
			continue
		}
		y0, x0, y1, x1 := d.decode(a, boxes[4*a:4*a+4])
		cand = append(cand, nms.Box{X0: x0, Y0: y0, X1: x1, Y1: y1, Score: s, Class: int32(best), Index: int32(a)})
	}
	return nms.Greedy(cand, iou, false, out), nil
}

// decode is TFLite's DecodeCenterSizeBoxes for anchor a: centre and half
// size in float64, corners in float32.
func (d *Decoder) decode(a int, e []int8) (y0, x0, y1, x1 float32) {
	an := d.cfg.Anchors[4*a : 4*a+4]
	ty, tx := float64(d.box[int(e[0])+128]), float64(d.box[int(e[1])+128])
	th, tw := float64(d.box[int(e[2])+128]), float64(d.box[int(e[3])+128])
	yc := float32(float64(ty/float64(d.cfg.YScale)*float64(an[2])) + float64(an[0]))
	xc := float32(float64(tx/float64(d.cfg.XScale)*float64(an[3])) + float64(an[1]))
	hh := float32(0.5 * math.Exp(th/float64(d.cfg.HScale)) * float64(an[2]))
	hw := float32(0.5 * math.Exp(tw/float64(d.cfg.WScale)) * float64(an[3]))
	return yc - hh, xc - hw, yc + hh, xc + hw
}
