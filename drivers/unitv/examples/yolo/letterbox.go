package main

import (
	"math"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// padValue fills the letterbox outside the image (YOLOX's preprocessing).
const padValue = 114

// letterbox converts a little-endian RGB565 camera image of w×h pixels to
// the detector's 416×416×3 int8 input the way YOLOX preprocesses: scaled by
// min(416/w, 416/h) with bilinear interpolation (half-pixel centres), placed
// at the top left, the rest filled with padValue, then − 128 (the model's
// cut QUANTIZE). Its tables are built once; run allocates nothing.
type letterbox struct {
	w, h   int
	scale  float32 // detector pixels per image pixel
	rw, rh int     // the scaled image inside the 416×416 input
	// Per output column (row): the two source columns (rows) and the
	// weight of the second, in 1/256.
	x0, x1, wx []uint16
	y0, y1, wy []uint16
}

func newLetterbox(w, h int) *letterbox {
	s := min(float32(inputSize)/float32(w), float32(inputSize)/float32(h))
	lb := &letterbox{w: w, h: h, scale: s,
		rw: min(inputSize, int(math.Round(float64(float32(w)*s)))),
		rh: min(inputSize, int(math.Round(float64(float32(h)*s))))}
	lb.x0, lb.x1, lb.wx = taps(lb.rw, w, s)
	lb.y0, lb.y1, lb.wy = taps(lb.rh, h, s)
	return lb
}

// taps returns, for n output positions over a source of size src scaled by
// s, the bilinear source pair and the second one's weight (of 256).
func taps(n, src int, s float32) (a, b, w []uint16) {
	a, b, w = make([]uint16, n), make([]uint16, n), make([]uint16, n)
	for i := range n {
		p := (float32(i)+0.5)/s - 0.5
		if p < 0 {
			p = 0
		}
		i0 := int(p)
		f := p - float32(i0)
		if i0 >= src-1 {
			i0, f = src-1, 0
		}
		a[i], b[i] = uint16(i0), uint16(min(i0+1, src-1))
		w[i] = uint16(f*256 + 0.5)
	}
	return a, b, w
}

// rgb unpacks pixel i of a little-endian RGB565 image to 8-bit channels.
func rgb(img []byte, i int) (r, g, b uint32) {
	p := uint32(img[2*i]) | uint32(img[2*i+1])<<8
	r, g, b = p>>11&31, p>>5&63, p&31
	return r<<3 | r>>2, g<<2 | g>>4, b<<3 | b>>2
}

// run converts img (w×h RGB565 LE) into in (416×416×3 int8).
func (lb *letterbox) run(in []int8, img []byte) {
	const pad = int8(padValue - 128)
	for y := range inputSize {
		row := in[y*inputSize*3 : (y+1)*inputSize*3]
		if y >= lb.rh {
			for i := range row {
				row[i] = pad
			}
			continue
		}
		top, bot := int(lb.y0[y])*lb.w, int(lb.y1[y])*lb.w
		wy := uint32(lb.wy[y])
		for x := range lb.rw {
			l, r := int(lb.x0[x]), int(lb.x1[x])
			wx := uint32(lb.wx[x])
			r00, g00, b00 := rgb(img, top+l)
			r01, g01, b01 := rgb(img, top+r)
			r10, g10, b10 := rgb(img, bot+l)
			r11, g11, b11 := rgb(img, bot+r)
			row[3*x] = mix(r00, r01, r10, r11, wx, wy)
			row[3*x+1] = mix(g00, g01, g10, g11, wx, wy)
			row[3*x+2] = mix(b00, b01, b10, b11, wx, wy)
		}
		for i := 3 * lb.rw; i < len(row); i++ {
			row[i] = pad
		}
	}
}

// mix interpolates four 8-bit values with weights of 256 and returns the
// result − 128.
func mix(p00, p01, p10, p11, wx, wy uint32) int8 {
	t := p00*(256-wx) + p01*wx
	b := p10*(256-wx) + p11*wx
	return int8(int32((t*(256-wy)+b*wy+32768)>>16) - 128)
}

// toImage turns a detection in detector pixels into a record entry in the
// camera image's pixels (score in Q0.16).
func (lb *letterbox) toImage(d nms.Box) frame.Detection {
	px := func(v float32) int16 {
		return int16(max(math.MinInt16, min(math.MaxInt16, math.Round(float64(v/lb.scale)))))
	}
	return frame.Detection{
		Class: uint8(d.Class),
		Score: uint16(max(0, min(65535, math.Round(float64(d.Score)*65535)))),
		X0:    px(d.X0), Y0: px(d.Y0), X1: px(d.X1), Y1: px(d.Y1),
	}
}
