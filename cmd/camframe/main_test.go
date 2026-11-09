package main

import (
	"bytes"
	"context"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

// recording returns a stream with device text around five images (seq 0..4):
// Y8, RGB565LE, JPEG, YUYV, RGB888Planar.
func recording(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	e := frame.NewEncoder(&buf)
	write := func(m frame.Meta, data []byte) {
		if err := e.WriteImage(m, 1000*e.Seq(), data); err != nil {
			t.Fatal(err)
		}
	}
	src := image.NewRGBA(image.Rect(0, 0, 32, 16))
	var jp bytes.Buffer
	if err := jpeg.Encode(&jp, src, nil); err != nil {
		t.Fatal(err)
	}
	buf.WriteString("boot banner\r\nsecond line\n")
	write(frame.Meta{Width: 160, Height: 120, Format: frame.Y8, Sensor: frame.SensorOV7740}, make([]byte, 160*120))
	buf.WriteString("log between\n")
	write(frame.Meta{Width: 320, Height: 240, Format: frame.RGB565LE}, make([]byte, 320*240*2))
	write(frame.Meta{Width: 32, Height: 16, Format: frame.JPEG, Sensor: frame.SensorOV2640}, jp.Bytes())
	write(frame.Meta{Width: 8, Height: 4, Format: frame.YUYV, Stride: 20}, make([]byte, 20*4))
	write(frame.Meta{Width: 5, Height: 3, Format: frame.RGB888Planar}, make([]byte, 45))
	buf.WriteString("tail without newline")
	return buf.Bytes()
}

func runFile(t *testing.T, n int, stream []byte) (dir, stdout, stderr string, err error) {
	t.Helper()
	dir = t.TempDir()
	in := filepath.Join(dir, "capture.bin")
	if err := os.WriteFile(in, stream, 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out")
	var so, se bytes.Buffer
	err = run(context.Background(), config{in: in, out: out, n: n, timeout: 10 * time.Second, raw: filepath.Join(dir, "raw.bin")}, &so, &se)
	return dir, so.String(), se.String(), err
}

func TestRunFileAll(t *testing.T) {
	stream := recording(t)
	dir, stdout, stderr, err := runFile(t, 0, stream)
	if err != nil {
		t.Fatalf("run: %v\nstdout:\n%s\nstderr:\n%s", err, stdout, stderr)
	}
	sizes := []image.Rectangle{
		image.Rect(0, 0, 160, 120), image.Rect(0, 0, 320, 240), image.Rect(0, 0, 32, 16),
		image.Rect(0, 0, 8, 4), image.Rect(0, 0, 5, 3),
	}
	for seq, want := range sizes {
		f, err := os.Open(filepath.Join(dir, "out", strconv.Itoa(seq)+".png"))
		if err != nil {
			t.Fatal(err)
		}
		img, err := png.Decode(f)
		f.Close()
		if err != nil || img.Bounds() != want {
			t.Fatalf("seq %d: %v bounds %v, want %v", seq, err, img.Bounds(), want)
		}
		if _, gray := img.(*image.Gray); gray != (seq == 0) {
			t.Errorf("seq %d PNG decodes as %T", seq, img)
		}
	}
	jpg, err := os.ReadFile(filepath.Join(dir, "out", "2.jpg"))
	if err != nil || !bytes.Contains(stream, jpg) || len(jpg) < 100 {
		t.Fatalf("raw jpg: %v len %d", err, len(jpg))
	}
	raw, err := os.ReadFile(filepath.Join(dir, "raw.bin"))
	if err != nil || !bytes.Equal(raw, stream) {
		t.Fatalf("raw tee: %v, %d of %d bytes", err, len(raw), len(stream))
	}
	wantErr := "dev| boot banner\ndev| second line\ndev| log between\ndev| tail without newline\n"
	if stderr != wantErr {
		t.Fatalf("stderr %q, want %q", stderr, wantErr)
	}
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) != 6 || !strings.HasPrefix(lines[0], "saved ") {
		t.Fatalf("stdout:\n%s", stdout)
	}
	summary := lines[5]
	textBytes := len("boot banner\r\nsecond line\n") + len("log between\n") + len("tail without newline")
	for _, want := range []string{"images=5 ", "saved=5 ", "fps=", "crc_errors=0 ", "dropped=0 ", "lost=0 ", "text_bytes=" + strconv.Itoa(textBytes)} {
		if !strings.Contains(summary, want) {
			t.Errorf("summary %q lacks %q", summary, want)
		}
	}
}

func TestRunFileStopsAtN(t *testing.T) {
	dir, stdout, _, err := runFile(t, 2, recording(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "out", "2.png")); err == nil {
		t.Fatal("saved a third image with -n 2")
	}
	if !strings.Contains(stdout, "images=2 saved=2 ") {
		t.Fatalf("stdout:\n%s", stdout)
	}
}

func TestRunFileTooFew(t *testing.T) {
	_, stdout, _, err := runFile(t, 9, recording(t))
	if err == nil || !strings.Contains(err.Error(), "saved 5 of 9 images: end of input") {
		t.Fatalf("err %v", err)
	}
	if !strings.Contains(stdout, "images=5 ") {
		t.Fatalf("stdout:\n%s", stdout)
	}
}

func TestParseFlags(t *testing.T) {
	cases := []struct {
		args []string
		ok   bool
	}{
		{[]string{"-in", "x.bin"}, true},
		{[]string{"-port", "/dev/null", "-baud", "921600", "-trigger", "c", "-n", "0"}, true},
		{[]string{}, false},
		{[]string{"-in", "x", "-port", "y"}, false},
		{[]string{"-in", "x", "-trigger", "c"}, false},
		{[]string{"-in", "x", "-n", "-1"}, false},
		{[]string{"-in", "x", "extra"}, false},
	}
	for _, tc := range cases {
		c, err := parseFlags(tc.args, io.Discard)
		if (err == nil) != tc.ok {
			t.Errorf("%q: err %v", tc.args, err)
		}
		if err == nil && (c.timeout != 30*time.Second || c.out != ".") && tc.args[0] == "-in" {
			t.Errorf("%q: defaults %+v", tc.args, c)
		}
	}
	c, _ := parseFlags([]string{"-in", "x"}, io.Discard)
	if c.baud != 115200 || c.n != 1 {
		t.Errorf("defaults %+v", c)
	}
}

func TestLineWriter(t *testing.T) {
	var out bytes.Buffer
	l := &lineWriter{w: &out, prefix: "dev| "}
	for _, chunk := range []string{"ab", "c\r\nde", "\x1b[31mf\n", "\tg"} {
		l.Write([]byte(chunk))
	}
	l.Flush()
	if want := "dev| abc\ndev| de.[31mf\ndev| \tg\n"; out.String() != want {
		t.Fatalf("got %q, want %q", out.String(), want)
	}
}
