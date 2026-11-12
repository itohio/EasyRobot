// Command qi8bench benchmarks the qi8 prepared layers at MobileNetV1-0.25
// 96×96 layer shapes — the shapes of qi8's BenchmarkMobileNet_* on the host —
// and the requantisation step alone, on whichever qi8 backend the build
// selects (ESP32-S3 PIE by default on the chip, generic with -tags
// qi8generic or on the host). It prints cycles (CCOUNT on the S3,
// nanoseconds on the host) per call, per MAC and per output, and an FNV of
// each kernel's output so that a change can be checked for bit-exactness
// against the previous run (ER-NN-PIPELINE step 3, §A0). See README.md.
package main

import (
	"time"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// xorshift fills s deterministically from seed.
func fill(s []int8, seed uint32) {
	x := seed*2654435761 + 1
	for i := range s {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		s[i] = int8(x)
	}
}

func fnv(b []int8) uint32 {
	h := uint32(2166136261)
	for _, v := range b {
		h ^= uint32(uint8(v))
		h *= 16777619
	}
	return h
}

func hex(v uint32) string {
	const d = "0123456789abcdef"
	var s [8]byte
	for i := 7; i >= 0; i-- {
		s[i] = d[v&15]
		v >>= 4
	}
	return string(s[:])
}

func requants(n int) []qi8.Requant {
	r := make([]qi8.Requant, n)
	for i := range r {
		r[i] = qi8.Requant{Mult: 1<<30 + int32(i)*12345, Shift: -8}
	}
	return r
}

type kernel struct {
	name     string
	op       qi8.Op
	src, dst []int8
	macs     int
}

// Shared input/output buffers: kernels run one at a time.
var srcBuf, dstBuf = make([]int8, 96*96*3), make([]int8, 48*48*16)

func conv(in qi8.Shape, outC, k, stride int, round qi8.Rounding) kernel {
	p := qi8.Conv2DParams{In: in, OutC: outC, KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame,
		InZP: -3, OutZP: 5, ActMin: -128, ActMax: 127, Round: round}
	os, _ := p.OutShape()
	l := &qi8.Conv2DLayer{P: p, W: make([]int8, outC*k*k*in.C), Bias: make([]int32, outC), RQ: requants(outC)}
	fill(l.W, 2)
	l.Scratch = make([]int8, l.ScratchLen())
	if err := l.Prepare(); err != nil {
		println("prepare:", err.Error())
	}
	name := "conv"
	if k == 1 {
		name = "pw"
	}
	return kernel{name, l, srcBuf[:in.Len()], dstBuf[:os.Len()], os.Len() * k * k * in.C}
}

func dw(in qi8.Shape, stride int, round qi8.Rounding) kernel {
	p := qi8.DepthwiseConv2DParams{In: in, KH: 3, KW: 3, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame,
		InZP: -3, OutZP: 5, ActMin: -128, ActMax: 127, Round: round}
	os, _ := p.OutShape()
	l := &qi8.DepthwiseConv2DLayer{P: p, W: make([]int8, 9*in.C), Bias: make([]int32, in.C), RQ: requants(in.C)}
	fill(l.W, 3)
	l.Scratch = make([]int8, l.ScratchLen())
	if err := l.Prepare(); err != nil {
		println("prepare:", err.Error())
	}
	return kernel{"dw", l, srcBuf[:in.Len()], dstBuf[:os.Len()], os.Len() * 9}
}

func kernels(round qi8.Rounding) []kernel {
	return []kernel{
		conv(qi8.Shape{H: 96, W: 96, C: 3}, 8, 3, 2, round),
		dw(qi8.Shape{H: 48, W: 48, C: 8}, 1, round),
		conv(qi8.Shape{H: 48, W: 48, C: 8}, 16, 1, 1, round),
		dw(qi8.Shape{H: 24, W: 24, C: 32}, 1, round),
		conv(qi8.Shape{H: 24, W: 24, C: 32}, 32, 1, 1, round),
		dw(qi8.Shape{H: 6, W: 6, C: 128}, 1, round),
		conv(qi8.Shape{H: 6, W: 6, C: 128}, 128, 1, 1, round),
	}
}

func main() {
	if device {
		time.Sleep(2 * time.Second) // let the USB console attach
	}
	rounds := []qi8.Rounding{qi8.RoundExact, qi8.RoundTFLite}
	sets := [][]kernel{kernels(rounds[0]), kernels(rounds[1])}
	acc := make([]int32, 4096)
	for i := range acc {
		acc[i] = int32(uint32(i)*2654435761) >> 12
	}
	rq := requants(16) // cycling multipliers, as per-channel requantisation sees them
	for rep := 0; ; rep++ {
		for ri, round := range rounds {
			println("=== qi8bench backend", qi8.Backend(), "round", int(round), "rep", rep, "unit", tickUnit)
			for i := range sets[ri] {
				k := &sets[ri][i]
				fill(k.src, uint32(i+7))
				if err := k.op.Run(k.dst, k.src); err != nil {
					println(k.name, "run:", err.Error())
					continue
				}
				h := fnv(k.dst)
				const n = 3
				t0 := stamp()
				for range n {
					_ = k.op.Run(k.dst, k.src)
				}
				t := ticksSince(t0) / n
				in, out := k.op.InShape(), k.op.OutShape()
				println(k.name, in.H, in.W, in.C, "->", out.H, out.W, out.C, "per_call", int(t),
					"per_MAC_x1000", int(t*1000/uint64(k.macs)), "per_out", int(t/uint64(len(k.dst))), "fnv", hex(h))
			}
			var s int32
			t0 := stamp()
			for i, a := range acc {
				r := rq[i&15]
				if round == qi8.RoundTFLite {
					s += r.ApplyTFLite(a)
				} else {
					s += int32(r.Apply(a))
				}
			}
			println("requant per_op_x10", int(ticksSince(t0)*10/uint64(len(acc))), "sum", int(s))
		}
		if !device {
			return
		}
		time.Sleep(2 * time.Second)
	}
}
