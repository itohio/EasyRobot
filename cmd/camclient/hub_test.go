package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// Snapshots: never more than keep of camclient's own snap-* files, oldest
// deleted; "_N" names of the same second count as newer than the plain name
// and order numerically; other files (grab's, evidence) are left alone.
func TestSnapshotRetention(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"snap-20261112-100000.jpg", "snap-20261112-100001.jpg", "notes.txt", "snap-20261112-1000.jpg", "20261112-090000.jpg", "photo20261114-120000.jpg"} {
		os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644)
	}
	now := time.Date(2026, 11, 12, 10, 0, 2, 0, time.Local)
	var paths []string
	for range 11 { // same second: plain, then _1 … _10
		p, err := saveSnapshot(dir, 4, []byte{0xFF, 0xD8}, now)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, filepath.Base(p))
	}
	if paths[0] != "snap-20261112-100002.jpg" || paths[10] != "snap-20261112-100002_10.jpg" {
		t.Errorf("names %v", paths)
	}
	ents, _ := os.ReadDir(dir)
	var got []string
	for _, e := range ents {
		got = append(got, e.Name())
	}
	// grab's YYYYmmdd-HHMMSS.jpg and anything else not snap-* stays (verify S2).
	want := []string{"20261112-090000.jpg", "notes.txt", "photo20261114-120000.jpg", "snap-20261112-1000.jpg", "snap-20261112-100002_10.jpg", "snap-20261112-100002_7.jpg", "snap-20261112-100002_8.jpg", "snap-20261112-100002_9.jpg"}
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Errorf("after 11 snapshots with keep 4: %v, want %v", got, want)
	}
}

// The page may send stream and control commands only: not G (a single
// image would interleave with the stream) and nothing that smuggles a
// second line.
func TestAllowedCmd(t *testing.T) {
	for line, want := range map[string]bool{
		"S": true, "X": true, "?": true, "Z 2": true, "Q 30": true, "E 500000": true, "E a": true, "I 1.5": true,
		"G 0 30": false, "G": false, "": false, "Z": false, "S 1": false, "E 1 2": false,
		"M 1 0": true, "M 1": false,
		"E 5\nS": false, "Q 30\r": false, "W 1": false,
	} {
		if got := allowedCmd(line); got != want {
			t.Errorf("allowedCmd(%q) = %v, want %v", line, got, want)
		}
	}
}

func TestLoopbackOnly(t *testing.T) {
	for addr, ok := range map[string]bool{
		"127.0.0.1:8765": true, "[::1]:8765": true, "127.0.0.2:80": true,
		"0.0.0.0:8765": false, ":8765": false, "192.168.1.5:8765": false, "localhost:8765": false, "127.0.0.1": false,
	} {
		if err := loopbackOnly(addr); (err == nil) != ok {
			t.Errorf("loopbackOnly(%q) = %v, want ok=%v", addr, err, ok)
		}
	}
}

// ioreg lists each USB device's properties, then its children; the callout
// device must come from the device with the wanted serial, not a later one.
func TestCalloutFor(t *testing.T) {
	out := []byte(`+-o Tufty 2040@01100000  <class IOUSBHostDevice>
    "USB Serial Number" = "E66164084319392A"
    +-o IOUSBHostInterface@0
      | "IOCalloutDevice" = "/dev/cu.usbmodem101"
+-o USB JTAG/serial debug unit@01200000  <class IOUSBHostDevice>
    "USB Serial Number" = "1C:DB:D4:74:55:80"
    +-o IOUSBHostInterface@0
      | "IOCalloutDevice" = "/dev/cu.usbmodem1101"
`)
	if got := calloutFor(out, "1C:DB:D4:74:55:80"); got != "/dev/cu.usbmodem1101" {
		t.Errorf("S3: %q", got)
	}
	if got := calloutFor(out, "E66164084319392A"); got != "/dev/cu.usbmodem101" {
		t.Errorf("badge: %q", got)
	}
	if got := calloutFor(out, "00:00"); got != "" {
		t.Errorf("absent serial: %q", got)
	}
	noTTY := []byte(`    "USB Serial Number" = "AA"
    "USB Serial Number" = "BB"
      | "IOCalloutDevice" = "/dev/cu.usbmodemBB"
`)
	if got := calloutFor(noTTY, "AA"); got != "" {
		t.Errorf("device without a tty took the next device's: %q", got)
	}
}

// Web Serial mode, settings: the page gets the owner's settings file as is,
// a 404 (it falls back to its defaults) when there is none, and "save
// settings" writes only valid, complete settings.
func TestSettingsRoute(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	h := newWebServer(config{settings: path}).routes()
	do := func(method, body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(method, "/settings.json", strings.NewReader(body)))
		return w
	}
	if w := do("GET", ""); w.Code != http.StatusNotFound {
		t.Errorf("no file: %d, want 404", w.Code)
	}
	if w := do("PUT", `{"size":2,"quality":12,"exposure":500000,"gain":1,"mirror":true,"flip":false}`); w.Code != 200 {
		t.Fatalf("valid PUT: %d %s", w.Code, w.Body)
	}
	saved := do("GET", "").Body.String()
	var back map[string]any
	if err := json.Unmarshal([]byte(saved), &back); err != nil || back["quality"] != 12.0 || back["exposure"] != 500000.0 || back["mirror"] != true {
		t.Errorf("saved %q (%v)", saved, err)
	}
	if w := do("PUT", `{"size":1,"quality":5,"exposure":"auto","gain":1,"wb":[1.47,1.02,1.31],"mirror":true,"flip":false}`); w.Code != 200 {
		t.Errorf("manual white balance: %d %s", w.Code, w.Body)
	}
	if got := do("GET", "").Body.String(); !strings.Contains(got, "1.47") || !strings.Contains(got, `"wb"`) {
		t.Errorf("white balance not saved: %q", got)
	}
	if w := do("PUT", `{"size":0,"quality":30,"exposure":"auto","gain":"auto","wb":"auto","mirror":true,"flip":true}`); w.Code != 200 {
		t.Errorf("auto values: %d %s", w.Code, w.Body)
	}
	autos := do("GET", "").Body.String()
	for _, bad := range []string{
		`{"size":5,"quality":30,"exposure":"auto","gain":1,"mirror":true,"flip":false}`,
		`{"size":0,"quality":0,"exposure":"auto","gain":1,"mirror":true,"flip":false}`,
		`{"size":0,"quality":30,"exposure":4000000,"gain":1,"mirror":true,"flip":false}`,
		`{"size":0,"quality":30,"exposure":"auto","gain":0.5,"mirror":true,"flip":false}`,
		`{"size":0,"quality":30,"exposure":"auto","gain":1,"mirror":true,"flip":false,"extra":1}`,
		`{"size":0,"quality":30,"exposure":"auto","gain":1,"wb":[1,1],"mirror":true,"flip":false}`,
		`{"size":0,"quality":30,"exposure":"auto","gain":1,"wb":[1,5,1],"mirror":true,"flip":false}`,
		`not json`,
		`{"size":0,"quality":30,"exposure":"auto","gain":1,"wb":[1,-0.1,1],"mirror":true,"flip":false}`,
		`{"size":0,"quality":30,"exposure":"auto","gain":1,"mirror":true,"flip":false}{"size":9}`,
	} {
		if w := do("PUT", bad); w.Code != http.StatusBadRequest {
			t.Errorf("PUT %s: %d, want 400", bad, w.Code)
		}
	}
	if got := do("GET", "").Body.String(); got != autos {
		t.Errorf("a rejected PUT changed the file: %q", got)
	}
}

// Web Serial mode, frame relay: peers get the page's latest image while it
// is connected, and a clear 503 otherwise (no stale frame).
func TestFrameRelay(t *testing.T) {
	dir := t.TempDir()
	s := newWebServer(config{settings: filepath.Join(dir, "settings.json"), out: dir, keep: 50})
	h := s.routes()
	req := func(method, path string, body []byte) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(method, path, bytes.NewReader(body)))
		return w
	}
	if w := req("POST", "/snapshot", nil); w.Code != http.StatusServiceUnavailable {
		t.Errorf("snapshot before any frame: %d, want 503", w.Code)
	}
	if w := req("POST", "/frame", []byte("hello")); w.Code != http.StatusBadRequest {
		t.Errorf("non-JPEG frame: %d, want 400", w.Code)
	}
	if w := req("POST", "/frame", []byte{0xFF, 0xD8, 1, 2, 3}); w.Code != http.StatusBadRequest {
		t.Errorf("JPEG without EOI: %d, want 400", w.Code)
	}
	huge := make([]byte, maxFrame+1)
	huge[0], huge[1], huge[maxFrame-1], huge[maxFrame] = 0xFF, 0xD8, 0xFF, 0xD9
	if w := req("POST", "/frame", huge); w.Code != http.StatusBadRequest {
		t.Errorf("1 MiB + 1 frame: %d, want 400", w.Code)
	}
	j := []byte{0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9}
	if w := req("POST", "/frame", j); w.Code != http.StatusNoContent {
		t.Fatalf("JPEG frame: %d", w.Code)
	}
	if w := req("GET", "/frame.jpg", nil); w.Code != 200 || !bytes.Equal(w.Body.Bytes(), j) {
		t.Errorf("frame.jpg: %d %x", w.Code, w.Body.Bytes())
	}
	w := req("POST", "/snapshot", nil)
	var out struct{ Path string }
	if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &out) != nil {
		t.Fatalf("snapshot: %d %s", w.Code, w.Body)
	}
	if b, err := os.ReadFile(out.Path); err != nil || !bytes.Equal(b, j) {
		t.Errorf("snapshot file %s: %x %v", out.Path, b, err)
	}
	if _, err := s.latest(time.Now().Add(staleAfter + time.Second)); err == nil {
		t.Error("a frame older than staleAfter is still served")
	}
}

// Both servers refuse a non-loopback Host (DNS rebinding) and a
// state-changing request from another site's page; the same-origin page and
// clients without an Origin (curl, peers) still work (verify S1).
func TestSameOriginGuard(t *testing.T) {
	dir := t.TempDir()
	web := sameOrigin(newWebServer(config{settings: filepath.Join(dir, "s.json"), out: dir, keep: 50}).routes())
	hold := sameOrigin(newHub(&bytes.Buffer{}, dir, 50).routes())
	jpeg := []byte{0xFF, 0xD8, 1, 0xFF, 0xD9}
	for _, tt := range []struct {
		name         string
		h            http.Handler
		method, path string
		host, origin string
		body         []byte
		want         int
	}{
		{"cross-site POST /cmd (-hold)", hold, "POST", "/cmd", "127.0.0.1:8765", "https://evil.example", []byte("c=E+3000000"), 403},
		{"cross-site POST /frame", web, "POST", "/frame", "127.0.0.1:8765", "https://evil.example", jpeg, 403},
		{"cross-site POST /snapshot", web, "POST", "/snapshot", "127.0.0.1:8765", "http://localhost:9999", nil, 403},
		{"rebound Host reads /frame.jpg", web, "GET", "/frame.jpg", "evil.example:8765", "", nil, 403},
		{"rebound Host loads the page", web, "GET", "/", "evil.example", "", nil, 403},
		{"non-loopback IP Host", web, "GET", "/frame.jpg", "192.168.1.5:8765", "", nil, 403},
		{"same-origin page POST /frame", web, "POST", "/frame", "127.0.0.1:8765", "http://127.0.0.1:8765", jpeg, 204},
		{"curl POST /frame (no Origin)", web, "POST", "/frame", "127.0.0.1:8765", "", jpeg, 204},
		{"localhost page", web, "GET", "/", "localhost:8765", "", nil, 200},
		{"[::1] page", web, "GET", "/", "[::1]:8765", "", nil, 200},
	} {
		r := httptest.NewRequest(tt.method, tt.path, bytes.NewReader(tt.body))
		r.Host = tt.host
		if tt.origin != "" {
			r.Header.Set("Origin", tt.origin)
		}
		if tt.method == "POST" && tt.path == "/cmd" {
			r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		}
		w := httptest.NewRecorder()
		tt.h.ServeHTTP(w, r)
		if w.Code != tt.want {
			t.Errorf("%s: %d, want %d", tt.name, w.Code, tt.want)
		}
	}
}

// Defaults are camclient's own, never the owner's folders or board
// (verify S2); -hold needs an explicit serial.
func TestDefaults(t *testing.T) {
	c, err := parseFlags(nil, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	cache, _ := os.UserCacheDir()
	conf, _ := os.UserConfigDir()
	if c.out != filepath.Join(cache, "camclient") || c.settings != filepath.Join(conf, "camclient", "settings.json") || c.serial != "" {
		t.Errorf("defaults out %q settings %q serial %q", c.out, c.settings, c.serial)
	}
	if _, err := parseFlags([]string{"-hold"}, io.Discard); err == nil {
		t.Error("-hold without -serial accepted")
	}
}
