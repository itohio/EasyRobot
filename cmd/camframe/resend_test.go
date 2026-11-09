package main

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

// lossyWriter drops writes of the first pass of an image: drop[img][k] drops
// the k-th (1-based) write of image img's WriteImage. Resends pass.
type lossyWriter struct {
	w       io.Writer
	img, n  int
	initial bool
	drop    map[int]map[int]bool
}

func (l *lossyWriter) Write(p []byte) (int, error) {
	if l.initial {
		l.n++
		if l.drop[l.img][l.n] {
			return len(p), nil
		}
	}
	return l.w.Write(p)
}

// sender is a device with chunk resend: it writes each image, then answers
// Naks with the chunk again until the Ack for that image.
func sender(t *testing.T, images int, ctl io.Reader, lossy *lossyWriter) {
	enc := frame.NewEncoder(lossy)
	data := make([]byte, 160*120)
	var p frame.ControlParser
	buf := make([]byte, 64)
	for i := range images {
		for j := range data {
			data[j] = byte(i + j)
		}
		lossy.img, lossy.n, lossy.initial = i, 0, true
		if err := enc.WriteImage(frame.Meta{Width: 160, Height: 120, Format: frame.Y8}, uint32(i*100_000), data); err != nil {
			t.Error(err)
			return
		}
		lossy.initial = false
		seq := enc.Seq() - 1
	wait:
		for {
			n, err := ctl.Read(buf)
			if err != nil {
				return
			}
			for _, b := range buf[:n] {
				c, ok := p.Feed(b)
				if !ok || c.Seq != seq {
					continue
				}
				if c.Kind == frame.Ack {
					break wait
				}
				if err := enc.Resend(int(c.Chunk)); err != nil {
					t.Error(err)
					return
				}
			}
		}
	}
}

func TestRunResendRecoversLostAndCorruptChunks(t *testing.T) {
	devR, devW := io.Pipe() // device → host
	ctlR, ctlW := io.Pipe() // host → device
	// WriteImage of a 5-chunk image makes 11 writes: header+meta+piece for
	// chunk 0, then header+piece pairs. Drop image 0's chunk 2 payload (write
	// 7: a CRC error) and image 1's whole last chunk (writes 10 and 11).
	lossy := &lossyWriter{w: devW, drop: map[int]map[int]bool{0: {7: true}, 1: {10: true, 11: true}}}
	go func() {
		sender(t, 3, ctlR, lossy)
		devW.Close()
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c := config{out: t.TempDir(), n: 3}
	dec := frame.NewDecoder(devR)
	st, err := runResend(ctx, c, dec, ctlW, io.Discard)
	ctlW.Close()
	if err != nil {
		t.Fatal(err)
	}
	if st.saved != 3 || st.lost != 0 || st.abandoned != 0 || st.gaveUp != 0 {
		t.Fatalf("stats %+v, want 3 saved, nothing lost or abandoned", st)
	}
	if st.naks < 2 {
		t.Fatalf("naks = %d, want ≥ 2 (one per lost chunk)", st.naks)
	}
	if got := dec.Stats().CRCErrors; got != 1 {
		t.Fatalf("CRC errors = %d, want 1 (the truncated chunk)", got)
	}
	if fps := st.fps(); fps < 9.9 || fps > 10.1 {
		t.Fatalf("fps from device timestamps = %.2f, want 10", fps)
	}
}
