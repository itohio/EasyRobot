package tfl

import (
	"encoding/binary"
	"math"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
)

// postProcessOp is the custom operator LoadDetection cuts.
const postProcessOp = "TFLite_Detection_PostProcess"

// PostProcess is the TFLite_Detection_PostProcess operator of an SSD export
// (TF Object Detection API), as LoadDetection reads it: the anchors and the
// operator's options. x/vision/detect/ssd decodes the graph's two outputs
// with them.
type PostProcess struct {
	// Anchors holds one [ycenter, xcenter, h, w] row per box, dequantised
	// as TFLite does ((q − zp)·scale in float32) when the tensor is uint8.
	Anchors []float32
	// MaxDetections caps the detections; MaxClassesPerDetection is the
	// classes reported a box (fast NMS); DetectionsPerClass (100 when
	// absent) applies to regular NMS.
	MaxDetections, MaxClassesPerDetection, DetectionsPerClass int
	// NumClasses excludes the background (column 0 of the scores).
	NumClasses int
	// ScoreThreshold and IoUThreshold are the NMS thresholds.
	ScoreThreshold, IoUThreshold float32
	// YScale, XScale, HScale, WScale divide the box encodings.
	YScale, XScale, HScale, WScale float32
	// RegularNMS is use_regular_nms (false when absent: the fast path).
	RegularNMS bool
}

// LoadDetection is Load for SSD exports ending in the custom
// TFLite_Detection_PostProcess operator (it must be the last one): the
// graph stops before it, its two activation inputs become Graph.Outputs
// (box encodings [anchors × 4], class scores [anchors × (1 + NumClasses)],
// in that order) and its anchors and options are returned. Everything else
// is as Load.
func LoadDetection(b []byte) (qnet.Graph, PostProcess, error) {
	m, err := Open(b)
	if err != nil {
		return qnet.Graph{}, PostProcess{}, err
	}
	c := converter{m: m, prod: make(map[int]int), detect: true}
	g, err := c.run()
	if err != nil {
		return qnet.Graph{}, PostProcess{}, err
	}
	return g, c.pp, nil
}

// postProcess converts the cut TFLite_Detection_PostProcess operator o:
// records its inputs as the graph outputs and reads anchors and options.
func (c *converter) postProcess(o *operator) error {
	if o.inputs.n != 3 {
		return c.fail("need box encodings, class scores and anchors")
	}
	// outs: the tensor indices of the box encodings and class scores.
	var outs, lens [2]int
	for k := range 2 {
		x, err := c.input(o, k)
		if err != nil {
			return err
		}
		_, outs[k], _ = c.tensorAt(o, k, false)
		lens[k] = x.shape.Len()
	}
	opts, err := flexNumbers(o.customOpts)
	if err != nil {
		return c.fail("custom options are not a FlexBuffers map")
	}
	need := func(key string) float64 {
		v, ok := opts[key]
		if !ok && err == nil {
			err = c.fail("missing option " + key)
		}
		return v
	}
	pp := PostProcess{
		MaxDetections:          int(need("max_detections")),
		MaxClassesPerDetection: int(need("max_classes_per_detection")),
		DetectionsPerClass:     100,
		NumClasses:             int(need("num_classes")),
		ScoreThreshold:         float32(need("nms_score_threshold")),
		IoUThreshold:           float32(need("nms_iou_threshold")),
		YScale:                 float32(need("y_scale")),
		XScale:                 float32(need("x_scale")),
		HScale:                 float32(need("h_scale")),
		WScale:                 float32(need("w_scale")),
		RegularNMS:             opts["use_regular_nms"] != 0,
	}
	if err != nil {
		return err
	}
	if v, ok := opts["detections_per_class"]; ok {
		pp.DetectionsPerClass = int(v)
	}
	if pp.MaxDetections <= 0 || pp.MaxClassesPerDetection <= 0 || pp.NumClasses <= 0 {
		return c.fail("max_detections, max_classes_per_detection and num_classes must be positive")
	}
	t, d, err := c.constant(o, 2)
	if err != nil {
		return err
	}
	if d == nil || len(t.shape) != 2 || t.shape[1] != 4 {
		return c.fail("anchors must be a constant [boxes, 4] tensor")
	}
	n := int(t.shape[0])
	if lens[0] != 4*n || lens[1] != (pp.NumClasses+1)*n {
		return c.fail("box encodings [boxes, 4] and class scores [boxes, 1 + num_classes] do not match the anchors")
	}
	pp.Anchors = make([]float32, 4*n)
	switch t.typ {
	case typeFloat32:
		if len(d) != 16*n {
			return ErrFormat
		}
		for i := range pp.Anchors {
			pp.Anchors[i] = math.Float32frombits(binary.LittleEndian.Uint32(d[4*i:]))
		}
	case typeUInt8:
		if len(d) != 4*n || t.scale.n != 1 || t.zp.n != 1 {
			return c.fail("uint8 anchors need the data and one scale and zero point")
		}
		s, zp := c.m.scaleAt(&t, 0), float32(c.m.zpAt(&t, 0))
		for i, q := range d {
			pp.Anchors[i] = (float32(q) - zp) * s
		}
	default:
		return c.failT(&t, "anchors must be float32 or uint8")
	}
	c.pp, c.cutOuts = pp, outs[:]
	return nil
}
