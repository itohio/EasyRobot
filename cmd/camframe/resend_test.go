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
	c := config{out: t.TempDir(), n: 3, baud: 1_500_000}
	dctx, stopDec := context.WithCancel(ctx)
	dec := frame.NewDecoder(&ctxReader{ctx: dctx, r: devR})
	st, err := runResend(ctx, c, dec, stopDec, ctlW, io.Discard)
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

// chattyDevice writes images through enc as fast as the pipe takes them
// and a log line between them, answering nothing; it never stops on its
// own, like a device that keeps streaming after the host is done.
func chattyDevice(w io.Writer, stop <-chan struct{}) {
	enc := frame.NewEncoder(w)
	data := make([]byte, 160*120)
	for i := 0; ; i++ {
		select {
		case <-stop:
			return
		default:
		}
		if _, err := io.WriteString(w, "device log line\r\n"); err != nil {
			return
		}
		if err := enc.WriteImage(frame.Meta{Width: 160, Height: 120, Format: frame.Y8}, uint32(i*100_000), data); err != nil {
			return
		}
	}
}

// TestRunResendStopsItsReaderBeforeReturning guards review M1: after
// runResend returns at -n, nothing may still touch the decoder or its Text
// writer, although the device keeps streaming. Run with -race.
func TestRunResendStopsItsReaderBeforeReturning(t *testing.T) {
	devR, devW := io.Pipe()
	stop := make(chan struct{})
	go chattyDevice(devW, stop)
	defer func() { close(stop); devR.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	dctx, stopDec := context.WithCancel(ctx)
	text := &lineWriter{w: io.Discard, prefix: "dev| "}
	dec := frame.NewDecoder(&ctxReader{ctx: dctx, r: newAsyncReader(ctx, devR)})
	dec.Text = text
	st, err := runResend(ctx, config{out: t.TempDir(), n: 3, baud: 1_500_000}, dec, stopDec, io.Discard, io.Discard)
	if err != nil || st.saved != 3 {
		t.Fatalf("saved %d, err %v", st.saved, err)
	}
	// The device is still writing; these must not race with a reader.
	for range 50 {
		text.Flush()
		_ = dec.Stats()
		time.Sleep(time.Millisecond)
	}
}

// pacedWriter delays each write by its airtime at baud (8N1).
type pacedWriter struct {
	w    io.Writer
	baud int
}

func (p pacedWriter) Write(b []byte) (int, error) {
	time.Sleep(time.Duration(len(b)) * 10 * time.Second / time.Duration(p.baud))
	return p.w.Write(b)
}

// TestRunResendCleanSlowLinkNeedsNoNaks guards review M2: on a lossless
// 115200 link a whole chunk takes ~0.36 s, and the idle timer must not Nak
// or give up while the image is still arriving.
func TestRunResendCleanSlowLinkNeedsNoNaks(t *testing.T) {
	if testing.Short() {
		t.Skip("takes ~4 s of simulated link time")
	}
	devR, devW := io.Pipe()
	ctlR, ctlW := io.Pipe()
	lossy := &lossyWriter{w: pacedWriter{w: devW, baud: 115200}}
	go func() {
		sender(t, 2, ctlR, lossy)
		devW.Close()
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	dctx, stopDec := context.WithCancel(ctx)
	dec := frame.NewDecoder(&ctxReader{ctx: dctx, r: devR})
	st, err := runResend(ctx, config{out: t.TempDir(), n: 2, baud: 115200}, dec, stopDec, ctlW, io.Discard)
	ctlW.Close()
	if err != nil {
		t.Fatal(err)
	}
	if st.saved != 2 || st.naks != 0 || st.gaveUp != 0 || st.abandoned != 0 {
		t.Fatalf("stats %+v, want 2 saved without Naks", st)
	}
}

func TestResendIdleScalesWithBaud(t *testing.T) {
	for _, tt := range []struct {
		baud     int
		min, max time.Duration
	}{
		{115200, 1100 * time.Millisecond, 1200 * time.Millisecond},  // 3 × 358 ms + 100 ms
		{1_500_000, 200 * time.Millisecond, 200 * time.Millisecond}, // 3 × 27 ms + 100 ms < floor
		{0, 200 * time.Millisecond, 200 * time.Millisecond},
	} {
		if got := resendIdle(tt.baud); got < tt.min || got > tt.max {
			t.Errorf("resendIdle(%d) = %v, want %v..%v", tt.baud, got, tt.min, tt.max)
		}
	}
}
