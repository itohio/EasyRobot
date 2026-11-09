// Command camframe fetches camera images in frame format v1
// (specified in the drivers/camera/frame package doc) from a serial port
// or a recorded stream file and writes them as PNG files.
//
//	camframe -port /dev/cu.usbserial-XXXX -baud 115200 -n 1 -out /tmp/cam
//	camframe -port /dev/cu.usbserial-XXXX -baud 1500000 -resend -n 100 -out /tmp/cam
//	camframe -in capture.bin -n 0 -out /tmp/cam
//
// Each image is written as <out>/<seq>.png; JPEG images are also saved raw as
// <out>/<seq>.jpg. Bytes between chunks (device logs) are printed to stderr
// prefixed "dev| ". One summary line is printed to stdout at the end.
//
// It opens the port with DTR and RTS deasserted. Without -resend it writes
// nothing to it except the -trigger bytes, once per wanted image. With
// -resend it Naks missing or corrupt chunks and Acks complete images
// (frame.Control), for senders that keep the image until the Ack.
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"image/png"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

type config struct {
	port    string
	baud    int
	in      string
	out     string
	n       int
	trigger string
	timeout time.Duration
	raw     string
	resend  bool
}

func parseFlags(args []string, errOut io.Writer) (config, error) {
	var c config
	fs := flag.NewFlagSet("camframe", flag.ContinueOnError)
	fs.SetOutput(errOut)
	fs.StringVar(&c.port, "port", "", "serial device, e.g. /dev/cu.usbserial-XXXX")
	fs.IntVar(&c.baud, "baud", 115200, "serial baud rate")
	fs.StringVar(&c.in, "in", "", "read a recorded stream file instead of a port")
	fs.StringVar(&c.out, "out", ".", "output directory")
	fs.IntVar(&c.n, "n", 1, "images to save; 0 = until interrupted, EOF or timeout")
	fs.StringVar(&c.trigger, "trigger", "", "if set, written to the port once per wanted image")
	fs.DurationVar(&c.timeout, "timeout", 30*time.Second, "overall timeout; 0 = none")
	fs.StringVar(&c.raw, "raw", "", "optional file receiving every received byte, for offline replay")
	fs.BoolVar(&c.resend, "resend", false, "chunk resend: Nak missing/corrupt chunks and Ack complete images (sender must support it; needs -port)")
	if err := fs.Parse(args); err != nil {
		return c, err
	}
	switch {
	case fs.NArg() != 0:
		return c, fmt.Errorf("unexpected arguments: %q", fs.Args())
	case (c.port == "") == (c.in == ""):
		return c, errors.New("exactly one of -port or -in is required")
	case c.trigger != "" && c.port == "":
		return c, errors.New("-trigger needs -port")
	case c.resend && c.port == "":
		return c, errors.New("-resend needs -port")
	case c.resend && c.trigger != "":
		return c, errors.New("-resend and -trigger are exclusive (resend paces by Ack)")
	case c.n < 0:
		return c, errors.New("-n must be ≥ 0")
	case c.timeout < 0:
		return c, errors.New("-timeout must be ≥ 0")
	}
	return c, nil
}

func main() {
	c, err := parseFlags(os.Args[1:], os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		return
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "camframe:", err)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, c, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "camframe:", err)
		stop()
		os.Exit(1)
	}
}

// run decodes images until c.n are saved, the input ends, the timeout
// expires or ctx is cancelled. It returns an error if fewer than c.n (> 0)
// images were saved.
func run(ctx context.Context, c config, stdout, stderr io.Writer) error {
	if err := os.MkdirAll(c.out, 0o755); err != nil {
		return err
	}

	if c.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	var src io.Reader
	var trig io.Writer
	if c.in != "" {
		f, err := os.Open(c.in)
		if err != nil {
			return err
		}
		defer f.Close()
		src = f
	} else {
		p, err := openPort(c.port, c.baud)
		if err != nil {
			return err
		}
		defer p.Close()
		src, trig = newAsyncReader(ctx, p), p
	}
	if c.raw != "" {
		f, err := os.Create(c.raw)
		if err != nil {
			return err
		}
		defer f.Close()
		src = &teeReader{r: src, w: f}
	}

	// The decoder reads through its own context, so a resend session can
	// stop and join its reader goroutine before the stats are read.
	dctx, stopDec := context.WithCancel(ctx)
	defer stopDec()
	text := &lineWriter{w: stderr, prefix: "dev| "}
	dec := frame.NewDecoder(&ctxReader{ctx: dctx, r: src})
	dec.Text = text

	if c.resend {
		st, err := runResend(ctx, c, dec, stopDec, trig, stdout)
		text.Flush()
		ds := dec.Stats()
		fmt.Fprintf(stdout, "images=%d saved=%d fps=%.2f crc_errors=%d naks=%d gave_up=%d abandoned=%d lost=%d text_bytes=%d\n",
			st.images, st.saved, st.fps(), ds.CRCErrors, st.naks, st.gaveUp, st.abandoned, st.lost, ds.TextBytes)
		return finish(c, st.saved, err)
	}
	var (
		saved       int
		first, last time.Time
		stopErr     error
	)
	for c.n == 0 || saved < c.n {
		if trig != nil && c.trigger != "" {
			if _, err := io.WriteString(trig, c.trigger); err != nil {
				stopErr = fmt.Errorf("write trigger: %w", err)
				break
			}
		}
		img, err := dec.Next()
		if err != nil {
			if err != io.EOF {
				stopErr = err
			}
			break
		}
		now := time.Now()
		if first.IsZero() {
			first = now
		}
		last = now
		if err := save(c.out, img, stdout); err != nil {
			fmt.Fprintf(stderr, "camframe: image %d: %v\n", img.Seq, err)
			continue
		}
		saved++
	}
	text.Flush()

	st := dec.Stats()
	fps := 0.0
	if st.Images > 1 && last.After(first) {
		fps = float64(st.Images-1) / last.Sub(first).Seconds()
	}
	fmt.Fprintf(stdout, "images=%d saved=%d fps=%.2f crc_errors=%d dropped=%d lost=%d text_bytes=%d\n",
		st.Images, saved, fps, st.CRCErrors, st.Dropped, st.Lost, st.TextBytes)

	return finish(c, saved, stopErr)
}

// finish turns the stop reason into run's error: none for a timeout or an
// interrupt unless fewer than c.n (> 0) images were saved.
func finish(c config, saved int, stopErr error) error {
	if stopErr != nil && stopErr != io.EOF && !errors.Is(stopErr, context.Canceled) && !errors.Is(stopErr, context.DeadlineExceeded) {
		return stopErr
	}
	if c.n > 0 && saved < c.n {
		reason := "end of input"
		switch {
		case errors.Is(stopErr, context.DeadlineExceeded):
			reason = fmt.Sprintf("timeout after %v", c.timeout)
		case errors.Is(stopErr, context.Canceled):
			reason = "interrupted"
		}
		return fmt.Errorf("saved %d of %d images: %s", saved, c.n, reason)
	}
	return nil
}

// save writes <dir>/<seq>.png, and for JPEG first the raw <dir>/<seq>.jpg.
func save(dir string, img frame.Image, stdout io.Writer) error {
	base := filepath.Join(dir, strconv.FormatUint(uint64(img.Seq), 10))
	if img.Meta.Format == frame.JPEG {
		if err := os.WriteFile(base+".jpg", img.Data, 0o644); err != nil {
			return err
		}
	}
	pic, err := toImage(img.Meta, img.Data)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, pic); err != nil {
		return err
	}
	if err := os.WriteFile(base+".png", buf.Bytes(), 0o644); err != nil {
		return err
	}
	m := img.Meta
	fmt.Fprintf(stdout, "saved %s.png %dx%d %s sensor=%s ts=%d bytes=%d\n",
		base, m.Width, m.Height, m.Format, m.Sensor, img.TS, m.DataLen)
	return nil
}

// contextReader is implemented by readers that can stop a blocked Read when
// a context ends.
type contextReader interface {
	readContext(ctx context.Context, p []byte) (int, error)
}

// ctxReader stops reading once ctx is done. A Read returning 0, nil (the
// serial port's poll timeout) is retried until data, an error or ctx ends.
// If r is a contextReader, a Read blocked inside it returns when ctx ends.
type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c *ctxReader) Read(p []byte) (int, error) {
	cr, _ := c.r.(contextReader)
	for {
		if err := c.ctx.Err(); err != nil {
			return 0, err
		}
		var n int
		var err error
		if cr != nil {
			n, err = cr.readContext(c.ctx, p)
		} else {
			n, err = c.r.Read(p)
		}
		if n > 0 || err != nil || len(p) == 0 {
			return n, err
		}
	}
}

// teeReader is io.TeeReader that keeps contextReader support.
type teeReader struct {
	r io.Reader
	w io.Writer
}

func (t *teeReader) Read(p []byte) (int, error) { return t.readContext(nil, p) }

func (t *teeReader) readContext(ctx context.Context, p []byte) (int, error) {
	var n int
	var err error
	if cr, ok := t.r.(contextReader); ok && ctx != nil {
		n, err = cr.readContext(ctx, p)
	} else {
		n, err = t.r.Read(p)
	}
	if n > 0 {
		if _, werr := t.w.Write(p[:n]); werr != nil {
			return n, werr
		}
	}
	return n, err
}

// asyncReader drains r in its own goroutine, so slow work on the decoding
// side (PNG encoding, disk writes, terminal output) never leaves the OS
// serial buffer unread. macOS drops serial input once that buffer fills:
// with a synchronous reader camframe lost 61–2,581-byte runs at 115200,
// while a tight reader on the same stream lost none
// (ER-K210-K5 evidence, outside the repo, K5b).
type asyncReader struct {
	ctx context.Context
	ch  chan []byte
	err error // set before ch is closed
	cur []byte
}

func newAsyncReader(ctx context.Context, r io.Reader) *asyncReader {
	a := &asyncReader{ctx: ctx, ch: make(chan []byte, 4096)} // up to 16 MiB queued
	go func() {
		for {
			b := make([]byte, 4096)
			n, err := r.Read(b)
			if n > 0 {
				select {
				case a.ch <- b[:n]:
				case <-ctx.Done():
				}
			}
			if err != nil || ctx.Err() != nil {
				if err == nil {
					err = ctx.Err()
				}
				a.err = err
				close(a.ch)
				return
			}
		}
	}()
	return a
}

func (a *asyncReader) Read(p []byte) (int, error) { return a.readContext(a.ctx, p) }

// readContext is Read that also returns when ctx ends.
func (a *asyncReader) readContext(ctx context.Context, p []byte) (int, error) {
	for len(a.cur) == 0 {
		select {
		case b, ok := <-a.ch:
			if !ok {
				if a.err == nil {
					return 0, io.EOF
				}
				return 0, a.err
			}
			a.cur = b
		case <-a.ctx.Done():
			return 0, a.ctx.Err()
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	}
	n := copy(p, a.cur)
	a.cur = a.cur[n:]
	return n, nil
}

// lineWriter writes device text line by line, each line prefixed, with a
// trailing CR removed and other control bytes (except tab) shown as '.'.
type lineWriter struct {
	w      io.Writer
	prefix string
	buf    []byte
}

// maxLine bounds a line without a newline (e.g. binary noise).
const maxLine = 4096

func (l *lineWriter) Write(p []byte) (int, error) {
	l.buf = append(l.buf, p...)
	for {
		i := bytes.IndexByte(l.buf, '\n')
		if i < 0 {
			if len(l.buf) < maxLine {
				return len(p), nil
			}
			i = len(l.buf)
		}
		l.emit(l.buf[:i])
		if i < len(l.buf) {
			i++
		}
		l.buf = l.buf[i:]
	}
}

// Flush writes a pending partial line.
func (l *lineWriter) Flush() {
	if len(l.buf) > 0 {
		l.emit(l.buf)
		l.buf = l.buf[:0]
	}
}

func (l *lineWriter) emit(line []byte) {
	line = bytes.TrimSuffix(line, []byte{'\r'})
	out := make([]byte, 0, len(l.prefix)+len(line)+1)
	out = append(out, l.prefix...)
	for _, b := range line {
		if (b < 0x20 && b != '\t') || b == 0x7F {
			b = '.'
		}
		out = append(out, b)
	}
	out = append(out, '\n')
	l.w.Write(out)
}
