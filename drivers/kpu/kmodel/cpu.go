package kmodel

import (
	"encoding/binary"
	"math"
	"unsafe"
)

// The CPU layers of the kmodel v3 runtime (kpu.c), on main memory main
// (MainMemUsage bytes) and, where a layer writes KPU memory, ram (RAMSize
// bytes). Floats in main memory are little-endian float32. Products are
// rounded to float32 before the add so that a compiler cannot fuse them
// into an FMA: results match the C runtime's float32 arithmetic.
//
// The layers assume the model passed Check.

func getF32(b []byte, i uint32) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
}

func putF32(b []byte, i uint32, v float32) {
	binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
}

// Dequantize writes float32(q)·Scale + Bias for Count uint8 values
// (kpu_kmodel_dequantize). The 256 possible results are computed once into
// lut (caller-owned, so that nothing is allocated per call), which on the
// K210 is cheaper than a float multiply-add per value.
func (q Quant) Dequantize(main []byte, lut *[256]uint32) {
	src, dst := main[q.In:q.In+q.Count], main[q.Out:q.Out+4*q.Count]
	for v := range lut {
		lut[v] = math.Float32bits(float32(float32(v)*q.Scale) + q.Bias)
	}
	if littleEndian && uintptr(unsafe.Pointer(unsafe.SliceData(dst)))%4 == 0 {
		d := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(dst))), len(src))
		for i, v := range src {
			d[i] = lut[v]
		}
		return
	}
	for i, v := range src {
		binary.LittleEndian.PutUint32(dst[4*i:], lut[v])
	}
}

// littleEndian reports whether native uint32 stores give the file's
// little-endian float layout (all of EasyRobot's targets; checked, not
// assumed, so that big-endian hosts still get correct bytes).
var littleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// Quantize writes clamp(round((f − Bias) / Scale), 0, 255) for Count float32
// values, rounding half away from zero like C roundf (kpu_quantize).
func (q Quant) Quantize(main []byte) {
	src, dst := main[q.In:], main[q.Out:q.Out+q.Count]
	inv := 1 / q.Scale
	for i := range dst {
		v := math.Round(float64((getF32(src, uint32(i)) - q.Bias) * inv))
		dst[i] = uint8(max(0, min(255, v)))
	}
}

// Run writes, per channel, the float32 mean of KernelSize consecutive
// float32 values, summed in order (kpu_global_average_pool2d).
func (g GAP) Run(main []byte) {
	src, dst := main[g.In:], main[g.Out:]
	k := uint32(0)
	for c := range g.Channels {
		var sum float32
		for range g.KernelSize {
			sum += getF32(src, k)
			k++
		}
		putF32(dst, c, sum/float32(g.KernelSize))
	}
}

// AddPadding copies Channels bytes from main memory into KPU memory as a
// 1×1 tensor in the 16-wide line layout the next conv reads
// (kpu_add_padding).
func (p Padding) AddPadding(main, ram []byte) {
	src := main[p.In : p.In+p.Channels]
	for c, v := range src {
		ram[padLayout.offset(p.Out, c, 0, 0)] = v
	}
}

// RemovePadding keeps the first byte of each channel's 16 (the 1×1 result
// of a conv run on a 4×4 padded input; kpu_remove_padding).
func (p Padding) RemovePadding(main []byte) {
	for c := range p.Channels {
		main[p.Out+c] = main[p.In+16*c]
	}
}

// Run copies the CHW tensor at In in main memory into KPU memory (kpu_upload).
func (u Upload) Run(main, ram []byte) error {
	w, h, c := int(u.Width), int(u.Height), int(u.Channels)
	return UploadLayout(w, h).UploadTo(ram, u.Out, w, h, c, main[u.In:])
}
