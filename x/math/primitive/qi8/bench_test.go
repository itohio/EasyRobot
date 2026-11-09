package qi8

import "testing"

func benchFill(n int, seed uint64) []int8 {
	g := newGen(seed)
	s := make([]int8, n)
	for i := range s {
		s[i] = int8(g.r.IntN(256) - 128)
	}
	return s
}

func benchRequants(n int) []Requant {
	rq := make([]Requant, n)
	for i := range rq {
		rq[i] = Requant{Mult: 1<<30 + int32(i)*12345, Shift: -8}
	}
	return rq
}

// reportMACs reports multiply-accumulates (nominal, including padded taps)
// per second.
func reportMACs(b *testing.B, macs int) {
	b.ReportMetric(float64(macs)*float64(b.N)/b.Elapsed().Seconds(), "MAC/s")
}

func benchConv(b *testing.B, in Shape, outC, k, stride int) {
	p := Conv2DParams{In: in, OutC: outC, KH: k, KW: k, StrideH: stride, StrideW: stride, InZP: -3, OutZP: 5, ActMin: -128, ActMax: 127}
	os, err := p.OutShape()
	if err != nil {
		b.Fatal(err)
	}
	src, w := benchFill(in.Len(), 1), benchFill(outC*k*k*in.C, 2)
	dst, bias, rq := make([]int8, os.Len()), make([]int32, outC), benchRequants(outC)
	b.ResetTimer()
	for b.Loop() {
		if err := Conv2D(dst, src, w, bias, rq, p); err != nil {
			b.Fatal(err)
		}
	}
	reportMACs(b, os.Len()*k*k*in.C)
}

func BenchmarkConv3x3_S1_16x16x16_16(b *testing.B) { benchConv(b, Shape{16, 16, 16}, 16, 3, 1) }
func BenchmarkConv3x3_S2_32x32x16_32(b *testing.B) { benchConv(b, Shape{32, 32, 16}, 32, 3, 2) }
func BenchmarkConv3x3_S1_32x32x3_16(b *testing.B)  { benchConv(b, Shape{32, 32, 3}, 16, 3, 1) }
func BenchmarkConv1x1_16x16x32_64(b *testing.B)    { benchConv(b, Shape{16, 16, 32}, 64, 1, 1) }

func benchDepthwise(b *testing.B, in Shape, stride int) {
	p := DepthwiseConv2DParams{In: in, KH: 3, KW: 3, StrideH: stride, StrideW: stride, InZP: -3, OutZP: 5, ActMin: -128, ActMax: 127}
	os, err := p.OutShape()
	if err != nil {
		b.Fatal(err)
	}
	src, w := benchFill(in.Len(), 1), benchFill(9*in.C, 2)
	dst, bias, rq := make([]int8, os.Len()), make([]int32, in.C), benchRequants(in.C)
	b.ResetTimer()
	for b.Loop() {
		if err := DepthwiseConv2D(dst, src, w, bias, rq, p); err != nil {
			b.Fatal(err)
		}
	}
	reportMACs(b, os.Len()*9)
}

func BenchmarkDepthwise3x3_S1_32x32x32(b *testing.B) { benchDepthwise(b, Shape{32, 32, 32}, 1) }
func BenchmarkDepthwise3x3_S2_32x32x32(b *testing.B) { benchDepthwise(b, Shape{32, 32, 32}, 2) }

func BenchmarkFullyConnected_1024x128(b *testing.B) {
	p := FCParams{In: 1024, Out: 128, InZP: -3, OutZP: 5, ActMin: -128, ActMax: 127}
	src, w := benchFill(p.In, 1), benchFill(p.In*p.Out, 2)
	dst, bias, rq := make([]int8, p.Out), make([]int32, p.Out), benchRequants(p.Out)
	for b.Loop() {
		if err := FullyConnected(dst, src, w, bias, rq, p); err != nil {
			b.Fatal(err)
		}
	}
	reportMACs(b, p.In*p.Out)
}

func BenchmarkDot_1024(b *testing.B) {
	x, y := benchFill(1024, 1), benchFill(1024, 2)
	var sink int32
	for b.Loop() {
		sink += Dot(x, y)
	}
	_ = sink
	reportMACs(b, 1024)
}

func benchPool(b *testing.B, f func(dst, src []int8, p Pool2DParams) error) {
	p := Pool2DParams{In: Shape{32, 32, 32}, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	os, _ := p.OutShape()
	src, dst := benchFill(p.In.Len(), 1), make([]int8, os.Len())
	for b.Loop() {
		if err := f(dst, src, p); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(p.In.Len())*float64(b.N)/b.Elapsed().Seconds(), "in-elem/s")
}

func BenchmarkMaxPool2x2_32x32x32(b *testing.B) { benchPool(b, MaxPool2D) }
func BenchmarkAvgPool2x2_32x32x32(b *testing.B) { benchPool(b, AvgPool2D) }

func BenchmarkReLU6_32K(b *testing.B) {
	src := benchFill(32768, 1)
	dst := make([]int8, len(src))
	for b.Loop() {
		_ = ReLU6(dst, src, -3, 90)
	}
	b.ReportMetric(float64(len(src))*float64(b.N)/b.Elapsed().Seconds(), "elem/s")
}
