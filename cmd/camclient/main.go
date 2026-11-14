// Command camclient is the badge-cam client for a XIAO ESP32-S3 Sense
// running the webcam example (drivers/esp32s3/xiaosense/examples/webcam).
// It serves a page named "client" on the loopback interface only.
//
//	camclient [-addr 127.0.0.1:8765] [-settings FILE] [-out DIR] [-keep 50]  # Web Serial (default)
//	camclient -hold -serial SERIAL [-addr …] [-out DIR] [-keep 50]
//
// Defaults are camclient's own: -settings <user config dir>/camclient/
// settings.json, -out <user cache dir>/camclient. In -out, camclient writes
// and prunes only its own snap-*.jpg files.
//
// Default (owner's choice): the page talks to the board itself over Web
// Serial (Chrome/Edge/Brave; 127.0.0.1 is a secure context), with a single
// Connect/Disconnect button: Connect opens the port, applies the settings
// and starts the stream; Disconnect stops it and releases the port, so grab
// works again. The page decodes the frame-v1 stream and shows the live
// image, fps, KB per frame, the board's exposure/gain and the failure
// counters; controls: exposure, gain, size, quality (1 … 63; a warning
// below 25), mirror (on by default)/flip, snapshot (download), settings
// (save to the -settings file, copy). camclient never opens the port. It
// serves (loopback only):
//
//	GET  /              the page
//	GET  /settings.json the -settings file
//	PUT  /settings.json validate and write it (the page's "save settings")
//	POST /frame         the page posts each image it receives
//	GET  /frame.jpg     the latest image while the page is connected
//	POST /snapshot      saves it to -out (snap-YYYYmmdd-HHMMSS.jpg, newest -keep kept); JSON {"path": …}
//
// -hold: camclient holds the port itself (found by USB serial number,
// opened exclusively without a DTR/RTS reset), keeps the stream running and
// serves an MJPEG page; for comparing stream rates with Web Serial. Ctrl-C
// stops the stream and releases the port. Its HTTP API:
//
//	GET  /            the page
//	GET  /stream.mjpg live images (multipart/x-mixed-replace)
//	GET  /frame.jpg   the latest image
//	GET  /status      JSON: host fps, image size, the board's status line
//	POST /cmd         form c=<command line>: Z, Q, E, I, M, S, X or ?
//	POST /snapshot    saves the latest image to -out; JSON {"path": …}
//	                  (snap-YYYYmmdd-HHMMSS.jpg, newest -keep kept)
//
// Both modes refuse requests whose Host is not a loopback name (DNS
// rebinding) and state-changing requests whose Origin is another site.
package main

import (
	"context"
	"embed"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

type config struct {
	hold     bool
	serial   string
	addr     string
	out      string
	keep     int
	settings string
}

//go:embed web
var webFiles embed.FS

func parseFlags(args []string, errOut io.Writer) (config, error) {
	fs := flag.NewFlagSet("camclient", flag.ContinueOnError)
	fs.SetOutput(errOut)
	var c config
	cache, _ := os.UserCacheDir()
	conf, _ := os.UserConfigDir()
	fs.BoolVar(&c.hold, "hold", false, "hold the serial port here and serve an MJPEG page (instead of Web Serial)")
	fs.StringVar(&c.settings, "settings", filepath.Join(conf, "camclient", "settings.json"), "settings file served as /settings.json")
	fs.StringVar(&c.serial, "serial", "", "USB serial number of the board (required with -hold)")
	fs.StringVar(&c.addr, "addr", "127.0.0.1:8765", "HTTP listen address (loopback only)")
	fs.StringVar(&c.out, "out", filepath.Join(cache, "camclient"), "snapshot directory (camclient prunes only its own snap-*.jpg there)")
	fs.IntVar(&c.keep, "keep", 50, "snapshots kept in -out (oldest deleted)")
	if err := fs.Parse(args); err != nil {
		return c, err
	}
	if err := loopbackOnly(c.addr); err != nil {
		return c, err
	}
	if c.keep < 1 || c.keep > 100 {
		return c, errors.New("-keep must be 1 … 100")
	}
	if c.hold && c.serial == "" {
		return c, errors.New("-hold needs -serial (the board's USB serial number)")
	}
	return c, nil
}

// loopbackOnly rejects listen addresses other than a loopback IP.
func loopbackOnly(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("-addr %q: %w", addr, err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("-addr %q: must be a loopback IP (127.0.0.1 or ::1)", addr)
	}
	return nil
}

func main() {
	c, err := parseFlags(os.Args[1:], os.Stderr)
	if err != nil {
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	run := runHold
	if !c.hold {
		run = serveWeb
	}
	if err := run(ctx, c); err != nil {
		fmt.Fprintln(os.Stderr, "camclient:", err)
		os.Exit(1)
	}
}

// serveWeb serves the Web Serial page, the settings file and the frame relay
// until ctx ends.
func serveWeb(ctx context.Context, c config) error {
	ln, err := net.Listen("tcp", c.addr)
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: sameOrigin(newWebServer(c).routes())}
	go srv.Serve(ln)
	fmt.Printf("client: http://%s/  (open in Chrome or Edge and press Connect; Ctrl-C stops)\n", ln.Addr())
	<-ctx.Done()
	sctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return srv.Shutdown(sctx)
}

// runHold holds the board's port, keeps the stream running and serves the
// MJPEG page until ctx ends.
func runHold(ctx context.Context, c config) error {
	path, err := findPort(c.serial)
	if err != nil {
		return err
	}
	p, err := openPort(path)
	if err != nil {
		return err
	}
	ln, err := net.Listen("tcp", c.addr)
	if err != nil {
		p.Close()
		return err
	}
	h := newHub(p, c.out, c.keep)
	readDone := make(chan error, 1)
	rctx, cancelRead := context.WithCancel(ctx)
	go func() { readDone <- h.readLoop(rctx) }()
	h.command("S")
	h.command("?")

	srv := &http.Server{Handler: sameOrigin(h.routes())}
	go srv.Serve(ln)
	fmt.Printf("client: http://%s/  (board %s on %s; Ctrl-C stops)\n", ln.Addr(), c.serial, path)

	var runErr error
	select {
	case <-ctx.Done():
	case runErr = <-readDone:
	}
	// Stop: end the stream so the board stops writing, then release the port.
	h.command("X")
	sctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	srv.Shutdown(sctx)
	cancel()
	cancelRead()
	if runErr == nil {
		runErr = <-readDone
	}
	if err := p.Close(); err != nil && runErr == nil {
		runErr = err
	}
	if errors.Is(runErr, context.Canceled) {
		runErr = nil
	}
	fmt.Println("client: stopped; port released")
	return runErr
}
