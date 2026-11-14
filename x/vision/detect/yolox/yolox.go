// Package yolox decodes the head outputs of an int8 YOLOX detector (anchor
// free, one prediction per grid cell and stride) into boxes, then runs
// greedy NMS. It suits exports whose box decoding is left out of the model:
// a scores tensor [anchors × classes] (objectness × class score) and a raw
// regression tensor [anchors × 4] (dx, dy, log w, log h), anchors ordered
// stride by stride (smallest first), row-major within a stride.
//
// After New, Decode allocates nothing: dequantisation and exp go through
// 256-entry tables and the score threshold is compared in the quantised
// domain.
package yolox

import (
	"errors"
	"math"

	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// Quant is a tensor's int8 quantisation: real = Scale · (q − ZP).
type Quant struct {
	Scale float32
	ZP    int32
}

// Config describes the model.
type Config struct {
	W, H    int   // input size in pixels
	Strides []int // e.g. 8, 16, 32
	Classes int
	Scores  Quant // scores tensor
	Boxes   Quant // regression tensor
}

// Decoder decodes one model's outputs. It is not safe for concurrent use.
type Decoder struct {
	cfg     Config
	gx, gy  []uint16 // grid cell of each anchor
	stride  []uint16
	lin     [256]float32 // Boxes.Scale·(q−ZP), indexed by q+128
	exp     [256]float32 // exp of the above
	cand    []nms.Box
	anchors int
}

// ErrConfig reports a configuration that cannot describe a YOLOX head.
var ErrConfig = errors.New("yolox: invalid config (W, H multiples of every stride, strides > 0, classes > 0, scales > 0)")

// New returns a Decoder for cfg holding at most maxCand candidates above
// the threshold per Decode (the rest, lowest scores, are dropped).
func New(cfg Config, maxCand int) (*Decoder, error) {
	if cfg.W <= 0 || cfg.H <= 0 || cfg.Classes <= 0 || len(cfg.Strides) == 0 || maxCand <= 0 ||
		!(cfg.Scores.Scale > 0) || !(cfg.Boxes.Scale > 0) {
		return nil, ErrConfig
	}
	d := &Decoder{cfg: cfg, cand: make([]nms.Box, 0, maxCand)}
	for _, s := range cfg.Strides {
		if s <= 0 || cfg.W%s != 0 || cfg.H%s != 0 {
			return nil, ErrConfig
		}
		for y := 0; y < cfg.H/s; y++ {
			for x := 0; x < cfg.W/s; x++ {
				d.gx = append(d.gx, uint16(x))
				d.gy = append(d.gy, uint16(y))
				d.stride = append(d.stride, uint16(s))
			}
		}
	}
	d.anchors = len(d.gx)
	for i := range 256 {
		v := cfg.Boxes.Scale * float32(int32(i-128)-cfg.Boxes.ZP)
		d.lin[i] = v
		d.exp[i] = float32(math.Exp(float64(v)))
	}
	return d, nil
}

// Anchors is the number of predictions (rows of both tensors).
func (d *Decoder) Anchors() int { return d.anchors }

// ErrShape reports output tensors of the wrong size.
var ErrShape = errors.New("yolox: scores or boxes tensor size does not match the config")

// Decode turns scores [Anchors × Classes] and boxes [Anchors × 4] (int8,
// quantised as the config says) into detections with score ≥ conf, keeps
// the best per class with greedy NMS at iou, and appends them to out (at
// most cap(out)), highest score first. Boxes are corners in input pixels.
func (d *Decoder) Decode(scores, boxes []int8, conf, iou float32, out []nms.Box) ([]nms.Box, error) {
	nc := d.cfg.Classes
	if len(scores) != d.anchors*nc || len(boxes) != d.anchors*4 {
		return out, ErrShape
	}
	// Smallest quantised score with Scale·(q−ZP) ≥ conf.
	tq := int32(math.Ceil(float64(conf/d.cfg.Scores.Scale))) + d.cfg.Scores.ZP
	if tq > 127 {
		return out, nil
	}
	thr := int8(max(tq, -128))
	cand := d.cand[:0]
	worst := -1 // index in cand of the lowest score, once cand is full
	for a := range d.anchors {
		row := scores[a*nc : a*nc+nc]
		best, cls := row[0], 0
		for c := 1; c < nc; c++ {
			if row[c] > best {
				best, cls = row[c], c
			}
		}
		if best < thr {
			continue
		}
		score := d.cfg.Scores.Scale * float32(int32(best)-d.cfg.Scores.ZP)
		if len(cand) == cap(cand) {
			if worst < 0 {
				worst = lowest(cand)
			}
			if score <= cand[worst].Score {
				continue
			}
		}
		b := boxes[a*4 : a*4+4]
		s := float32(d.stride[a])
		cx := (d.lin[int(b[0])+128] + float32(d.gx[a])) * s
		cy := (d.lin[int(b[1])+128] + float32(d.gy[a])) * s
		hw := d.exp[int(b[2])+128] * s / 2
		hh := d.exp[int(b[3])+128] * s / 2
		box := nms.Box{X0: cx - hw, Y0: cy - hh, X1: cx + hw, Y1: cy + hh, Score: score, Class: int32(cls), Index: int32(a)}
		if len(cand) < cap(cand) {
			cand = append(cand, box)
			continue
		}
		cand[worst] = box
		worst = lowest(cand)
	}
	d.cand = cand
	return nms.Greedy(cand, iou, true, out), nil
}

func lowest(c []nms.Box) int {
	w := 0
	for i := range c {
		if c[i].Score < c[w].Score {
			w = i
		}
	}
	return w
}
