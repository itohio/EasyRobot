package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

// hub holds the latest image and the board's status and fans images out to
// the live-view connections.
type hub struct {
	port io.ReadWriter
	out  string
	keep int

	wmu sync.Mutex // serialises command writes

	mu      sync.Mutex
	jpeg    []byte
	meta    frame.Meta
	seq     uint32
	frames  uint64
	last    time.Time
	fps     float64           // host-measured, smoothed
	status  map[string]string // the board's last status line
	lastErr string            // the board's last error line
	update  chan struct{}     // closed and replaced on every new image
}

func newHub(port io.ReadWriter, out string, keep int) *hub {
	return &hub{port: port, out: out, keep: keep, status: map[string]string{}, update: make(chan struct{})}
}

// command writes one command line to the board.
func (h *hub) command(line string) error {
	h.wmu.Lock()
	defer h.wmu.Unlock()
	_, err := io.WriteString(h.port, line+"\n")
	return err
}

// readLoop decodes images and text lines until ctx ends or the port fails.
func (h *hub) readLoop(ctx context.Context) error {
	d := frame.NewDecoder(&waitReader{ctx: ctx, r: h.port})
	d.Text = &lineSplitter{fn: h.textLine}
	for {
		img, err := d.Next()
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("board stream: %w", err)
		}
		if img.Meta.Format != frame.JPEG {
			continue
		}
		h.setImage(img, time.Now())
	}
}

func (h *hub) setImage(img frame.Image, now time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.last.IsZero() {
		if dt := now.Sub(h.last).Seconds(); dt > 0 {
			h.fps += (1/dt - h.fps) / 4
		}
	}
	h.jpeg, h.meta, h.seq, h.last = img.Data, img.Meta, img.Seq, now
	h.frames++
	close(h.update)
	h.update = make(chan struct{})
}

// textLine takes one text line from the board: status lines replace the
// status, error lines are kept for the page.
func (h *hub) textLine(line string) {
	// Bytes of a torn chunk can precede a line: start at the board's prefix.
	if i := strings.Index(line, "webcam: "); i > 0 {
		line = line[i:]
	}
	switch {
	case strings.HasPrefix(line, "webcam: status "):
		st := parseStatus(strings.TrimPrefix(line, "webcam: status "))
		h.mu.Lock()
		h.status = st
		if st["stream"] == "0" {
			h.fps = 0
		}
		h.mu.Unlock()
	case strings.HasPrefix(line, "webcam: error"):
		fmt.Fprintln(os.Stderr, "board:", strings.TrimPrefix(line, "webcam: "))
		h.mu.Lock()
		h.lastErr = strings.TrimPrefix(line, "webcam: ")
		h.mu.Unlock()
	}
}

// parseStatus splits "k=v k=v …" into a map.
func parseStatus(s string) map[string]string {
	m := map[string]string{}
	for _, f := range strings.Fields(s) {
		if k, v, ok := strings.Cut(f, "="); ok {
			m[k] = v
		}
	}
	return m
}

func (h *hub) latest() ([]byte, uint32, chan struct{}) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.jpeg, h.seq, h.update
}

func (h *hub) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		io.WriteString(w, page)
	})
	mux.HandleFunc("GET /frame.jpg", h.serveFrame)
	mux.HandleFunc("GET /stream.mjpg", h.serveStream)
	mux.HandleFunc("GET /status", h.serveStatus)
	mux.HandleFunc("POST /cmd", h.serveCmd)
	mux.HandleFunc("POST /snapshot", h.serveSnapshot)
	return mux
}

func (h *hub) serveFrame(w http.ResponseWriter, r *http.Request) {
	j, _, _ := h.latest()
	if j == nil {
		http.Error(w, "no image yet", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(j)
}

const boundary = "camclientframe"

func (h *hub) serveStream(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary="+boundary)
	w.Header().Set("Cache-Control", "no-store")
	fl, _ := w.(http.Flusher)
	var sent uint32
	first := true
	for {
		j, seq, upd := h.latest()
		if j != nil && (first || seq != sent) {
			if _, err := fmt.Fprintf(w, "--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", boundary, len(j)); err != nil {
				return
			}
			if _, err := w.Write(j); err != nil {
				return
			}
			if _, err := io.WriteString(w, "\r\n"); err != nil {
				return
			}
			if fl != nil {
				fl.Flush()
			}
			sent, first = seq, false
		}
		select {
		case <-upd:
		case <-r.Context().Done():
			return
		}
	}
}

type statusJSON struct {
	FPS    float64           `json:"fps"`
	Width  uint16            `json:"width"`
	Height uint16            `json:"height"`
	Bytes  int               `json:"bytes"`
	Frames uint64            `json:"frames"`
	AgeMS  int64             `json:"age_ms"`
	Board  map[string]string `json:"board"`
	Error  string            `json:"error,omitempty"`
}

func (h *hub) serveStatus(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	s := statusJSON{FPS: h.fps, Width: h.meta.Width, Height: h.meta.Height, Bytes: len(h.jpeg), Frames: h.frames, Board: h.status, Error: h.lastErr}
	if !h.last.IsZero() {
		s.AgeMS = time.Since(h.last).Milliseconds()
	}
	h.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s)
}

// allowedCmd reports whether line is a command the page may send: the
// stream and control commands, not G (single image), which would interleave
// with the stream.
func allowedCmd(line string) bool {
	f := strings.Fields(line)
	if len(f) == 0 || strings.ContainsAny(line, "\r\n") {
		return false
	}
	switch f[0] {
	case "S", "X", "?":
		return len(f) == 1
	case "Z", "Q", "E", "I":
		return len(f) == 2
	case "M":
		return len(f) == 3
	}
	return false
}

func (h *hub) serveCmd(w http.ResponseWriter, r *http.Request) {
	line := strings.TrimSpace(r.FormValue("c"))
	if !allowedCmd(line) {
		http.Error(w, "unknown command", http.StatusBadRequest)
		return
	}
	h.mu.Lock()
	h.lastErr = ""
	h.mu.Unlock()
	if err := h.command(line); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *hub) serveSnapshot(w http.ResponseWriter, r *http.Request) {
	j, _, _ := h.latest()
	if j == nil {
		http.Error(w, "no image yet", http.StatusServiceUnavailable)
		return
	}
	path, err := saveSnapshot(h.out, h.keep, j, time.Now())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"path": path})
}

// snapPrefix marks the files camclient writes; prune touches nothing else
// (grab's YYYYmmdd-HHMMSS.jpg files and evidence in the same folder stay).
const snapPrefix = "snap-"

// isSnapshotName matches snap-YYYYmmdd-HHMMSS.jpg and snap-YYYYmmdd-HHMMSS_N.jpg.
func isSnapshotName(n string) bool {
	if !strings.HasPrefix(n, snapPrefix) {
		return false
	}
	n = n[len(snapPrefix):]
	if len(n) < 19 || !strings.HasSuffix(n, ".jpg") || n[8] != '-' {
		return false
	}
	for i, c := range n[:15] {
		if i != 8 && (c < '0' || c > '9') {
			return false
		}
	}
	rest := n[15 : len(n)-4]
	return rest == "" || rest[0] == '_' && len(rest) > 1 && strings.Trim(rest[1:], "0123456789") == ""
}

// saveSnapshot writes j as dir/snap-<timestamp>.jpg, or snap-<timestamp>_N.jpg with
// N above any taken in the same second (so a newer snapshot never sorts
// before an older one, even after pruning), and deletes the oldest
// snapshots beyond keep.
func saveSnapshot(dir string, keep int, j []byte, now time.Time) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	stamp := snapPrefix + now.Format("20060102-150405")
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	next := 0 // 0: the plain name is free and no _N exists for this second
	for _, e := range ents {
		n := e.Name()
		if !isSnapshotName(n) || n[:len(stamp)] != stamp {
			continue
		}
		k := 1
		if rest := n[len(stamp) : len(n)-4]; rest != "" {
			k, _ = strconv.Atoi(rest[1:])
			k++
		}
		next = max(next, k)
	}
	path := filepath.Join(dir, stamp+".jpg")
	if next > 0 {
		path = filepath.Join(dir, fmt.Sprintf("%s_%d.jpg", stamp, next))
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return "", err
	}
	_, err = f.Write(j)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", err
	}
	return path, prune(dir, keep)
}

// prune keeps the newest keep snapshot files in dir. Names sort by time;
// "_N" suffixes sort after the plain name of the same second.
func prune(dir string, keep int) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var names []string
	for _, e := range ents {
		if !e.IsDir() && isSnapshotName(e.Name()) {
			names = append(names, e.Name())
		}
	}
	sort.Slice(names, func(i, j int) bool { return snapshotKey(names[i]) < snapshotKey(names[j]) })
	for _, n := range names[:max(0, len(names)-keep)] {
		if err := os.Remove(filepath.Join(dir, n)); err != nil {
			return err
		}
	}
	return nil
}

// snapshotKey orders names by second, then by _N numerically.
func snapshotKey(n string) string {
	n = n[len(snapPrefix):]
	rest := n[15 : len(n)-4]
	if rest == "" {
		return n[:15] + "_000000"
	}
	return n[:15] + "_" + strings.Repeat("0", max(0, 6-len(rest[1:]))) + rest[1:]
}

// waitReader turns the port's 100 ms empty reads into waiting, so the
// decoder's buffered reader never sees a run of empty reads; it returns
// io.EOF once ctx ends.
type waitReader struct {
	ctx context.Context
	r   io.Reader
}

func (w *waitReader) Read(p []byte) (int, error) {
	for {
		if w.ctx.Err() != nil {
			return 0, io.EOF
		}
		n, err := w.r.Read(p)
		if n > 0 || err != nil {
			return n, err
		}
	}
}

// lineSplitter passes complete text lines (without CR/LF) to fn.
type lineSplitter struct {
	buf []byte
	fn  func(string)
}

func (l *lineSplitter) Write(p []byte) (int, error) {
	for _, b := range p {
		if b == '\n' || b == '\r' {
			if len(l.buf) > 0 {
				l.fn(string(l.buf))
				l.buf = l.buf[:0]
			}
			continue
		}
		if len(l.buf) < 512 {
			l.buf = append(l.buf, b)
		}
	}
	return len(p), nil
}
