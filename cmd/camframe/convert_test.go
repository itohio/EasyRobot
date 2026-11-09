package main

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

func meta(f frame.Format, w, h, stride, n int) frame.Meta {
	return frame.Meta{Width: uint16(w), Height: uint16(h), Format: f, Stride: uint16(stride), DataLen: uint32(n)}
}

func rgbaAt(t *testing.T, img image.Image, x, y int) [3]uint8 {
	t.Helper()
	c := color.RGBAModel.Convert(img.At(x, y)).(color.RGBA)
	if c.A != 0xFF {
		t.Fatalf("pixel (%d,%d) alpha %d", x, y, c.A)
	}
	return [3]uint8{c.R, c.G, c.B}
}

func TestRGB565ByteOrder(t *testing.T) {
	cases := []struct {
		name   string
		bytes  [2]byte
		le, be [3]uint8
	}{
		{"red LE", [2]byte{0x00, 0xF8}, [3]uint8{255, 0, 0}, [3]uint8{0, 28, 198}},
		{"green LE", [2]byte{0xE0, 0x07}, [3]uint8{0, 255, 0}, [3]uint8{231, 0, 57}},
		{"mid", [2]byte{0x10, 0x84}, [3]uint8{132, 130, 132}, [3]uint8{16, 16, 33}},
		{"blue LE", [2]byte{0x1F, 0x00}, [3]uint8{0, 0, 255}, [3]uint8{24, 227, 0}},
		{"white", [2]byte{0xFF, 0xFF}, [3]uint8{255, 255, 255}, [3]uint8{255, 255, 255}},
		{"black", [2]byte{0x00, 0x00}, [3]uint8{0, 0, 0}, [3]uint8{0, 0, 0}},
	}
	for _, tc := range cases {
		for _, f := range []frame.Format{frame.RGB565LE, frame.RGB565BE} {
			img, err := toImage(meta(f, 1, 1, 0, 2), tc.bytes[:])
			if err != nil {
				t.Fatal(err)
			}
			want := tc.le
			if f == frame.RGB565BE {
				want = tc.be
			}
			if got := rgbaAt(t, img, 0, 0); got != want {
				t.Errorf("%s as %s: got %v, want %v", tc.name, f, got, want)
			}
		}
	}
}

// bt601 is the BT.601 full-range reference conversion.
func bt601(y, u, v float64) [3]uint8 {
	c := func(x float64) uint8 { return uint8(math.Max(0, math.Min(255, math.Round(x)))) }
	return [3]uint8{
		c(y + 1.402*(v-128)),
		c(y - 0.344136*(u-128) - 0.714136*(v-128)),
		c(y + 1.772*(u-128)),
	}
}

func TestYUYV(t *testing.T) {
	cases := []struct {
		name    string
		y, u, v uint8
		want    [3]uint8 // exact expectation; zero means "use the reference ±1"
		exact   bool
	}{
		{"black", 0, 128, 128, [3]uint8{0, 0, 0}, true},
		{"grey", 128, 128, 128, [3]uint8{128, 128, 128}, true},
		{"white", 255, 128, 128, [3]uint8{255, 255, 255}, true},
		{"red", 76, 85, 255, [3]uint8{}, false},
		{"green", 150, 44, 21, [3]uint8{}, false},
		{"blue", 29, 255, 107, [3]uint8{}, false},
		{"orange", 160, 60, 190, [3]uint8{}, false},
	}
	for _, tc := range cases {
		// Pixel 0 uses Y0 = tc.y, pixel 1 uses Y1 = 255 - tc.y; both share U, V.
		y1 := 255 - tc.y
		img, err := toImage(meta(frame.YUYV, 2, 1, 0, 4), []byte{tc.y, tc.u, y1, tc.v})
		if err != nil {
			t.Fatal(err)
		}
		for x, yy := range []uint8{tc.y, y1} {
			want := bt601(float64(yy), float64(tc.u), float64(tc.v))
			if tc.exact && x == 0 {
				want = tc.want
			}
			got := rgbaAt(t, img, x, 0)
			for i := range 3 {
				if d := int(got[i]) - int(want[i]); d < -1 || d > 1 {
					t.Errorf("%s pixel %d: got %v, want %v", tc.name, x, got, want)
					break
				}
			}
		}
	}
	// Saturated red must actually look red.
	img, _ := toImage(meta(frame.YUYV, 2, 1, 0, 4), []byte{76, 85, 76, 255})
	if c := rgbaAt(t, img, 0, 0); c[0] < 250 || c[1] > 5 || c[2] > 5 {
		t.Errorf("red: %v", c)
	}
}

func TestRGB888PlanarOrder(t *testing.T) {
	tight := []byte{
		1, 2, 3, 4, // R plane
		11, 12, 13, 14, // G plane
		21, 22, 23, 24, // B plane
	}
	padded := []byte{
		1, 2, 0xEE, 3, 4, 0xEE, // R plane rows, 3 bytes each
		11, 12, 0xEE, 13, 14, 0xEE,
		21, 22, 0xEE, 23, 24, 0xEE,
	}
	for _, tc := range []struct {
		name   string
		stride int
		data   []byte
	}{{"tight", 0, tight}, {"stride 9", 9, padded}} {
		img, err := toImage(meta(frame.RGB888Planar, 2, 2, tc.stride, len(tc.data)), tc.data)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		want := map[[2]int][3]uint8{{0, 0}: {1, 11, 21}, {1, 0}: {2, 12, 22}, {0, 1}: {3, 13, 23}, {1, 1}: {4, 14, 24}}
		for p, w := range want {
			if got := rgbaAt(t, img, p[0], p[1]); got != w {
				t.Errorf("%s (%d,%d): got %v, want %v", tc.name, p[0], p[1], got, w)
			}
		}
	}
}

func TestStridePaddingIgnored(t *testing.T) {
	y8, err := toImage(meta(frame.Y8, 3, 2, 5, 10), []byte{1, 2, 3, 99, 99, 4, 5, 6, 99, 99})
	if err != nil {
		t.Fatal(err)
	}
	g, ok := y8.(*image.Gray)
	if !ok || g.Bounds() != image.Rect(0, 0, 3, 2) {
		t.Fatalf("Y8 gives %T %v", y8, y8.Bounds())
	}
	for i, want := range []uint8{1, 2, 3, 4, 5, 6} {
		if got := g.GrayAt(i%3, i/3).Y; got != want {
			t.Errorf("Y8 pixel %d = %d, want %d", i, got, want)
		}
	}

	// RGB565LE 1×2 with stride 4: the pad word (white) must not show.
	rgb, err := toImage(meta(frame.RGB565LE, 1, 2, 4, 8), []byte{0x00, 0xF8, 0xFF, 0xFF, 0x1F, 0x00, 0xFF, 0xFF})
	if err != nil {
		t.Fatal(err)
	}
	if rgb.Bounds() != image.Rect(0, 0, 1, 2) || rgbaAt(t, rgb, 0, 0) != [3]uint8{255, 0, 0} || rgbaAt(t, rgb, 0, 1) != [3]uint8{0, 0, 255} {
		t.Errorf("RGB565 stride: %v %v", rgbaAt(t, rgb, 0, 0), rgbaAt(t, rgb, 0, 1))
	}

	// YUYV 2×2 with stride 6: the pad bytes are 0 (would be dark green if used).
	yuyv, err := toImage(meta(frame.YUYV, 2, 2, 6, 12), []byte{200, 128, 200, 128, 0, 0, 100, 128, 100, 128, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if rgbaAt(t, yuyv, 1, 0) != [3]uint8{200, 200, 200} || rgbaAt(t, yuyv, 1, 1) != [3]uint8{100, 100, 100} {
		t.Errorf("YUYV stride: %v %v", rgbaAt(t, yuyv, 1, 0), rgbaAt(t, yuyv, 1, 1))
	}
}

func TestJPEGDecode(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 16, 8))
	for i := range src.Pix {
		src.Pix[i] = 0x80
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, nil); err != nil {
		t.Fatal(err)
	}
	img, err := toImage(meta(frame.JPEG, 16, 8, 0, buf.Len()), buf.Bytes())
	if err != nil || img.Bounds() != src.Bounds() {
		t.Fatalf("JPEG: %v %v", err, img)
	}
	if _, err := toImage(meta(frame.JPEG, 16, 8, 0, 3), []byte{1, 2, 3}); err == nil {
		t.Fatal("garbage JPEG decoded")
	}
}

func TestConvertErrors(t *testing.T) {
	if _, err := toImage(meta(frame.Y8, 4, 4, 0, 15), make([]byte, 15)); err == nil {
		t.Error("short Y8 accepted")
	}
	if _, err := toImage(meta(frame.Y8, 4, 4, 0, 16), make([]byte, 15)); err == nil {
		t.Error("data shorter than meta accepted")
	}
	if _, err := toImage(meta(9, 4, 4, 0, 16), make([]byte, 16)); err == nil {
		t.Error("unknown format accepted")
	}
}
