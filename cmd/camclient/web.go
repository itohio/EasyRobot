package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"sync"
	"time"
)

// webServer backs the Web Serial page. The page holds the port; it posts
// every image it receives to /frame, so peers still get the live image
// (/frame.jpg, /snapshot) while the owner is connected. The page also saves
// the owner's settings to the settings file grab applies.
type webServer struct {
	settings string
	out      string
	keep     int

	mu   sync.Mutex
	jpeg []byte
	at   time.Time
	seq  uint64
}

// maxFrame bounds a posted image: a 2048×1536 JPEG buffer is 790,272 B.
const maxFrame = 1 << 20

// staleAfter is how long after the last posted image /snapshot still
// answers: longer than a 3 s exposure plus transfer.
const staleAfter = 5 * time.Second

func newWebServer(c config) *webServer {
	return &webServer{settings: c.settings, out: c.out, keep: c.keep}
}

func (s *webServer) routes() http.Handler {
	sub, _ := fs.Sub(webFiles, "web") // cannot fail: web is embedded
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(sub))
	mux.HandleFunc("GET /settings.json", s.getSettings)
	mux.HandleFunc("PUT /settings.json", s.putSettings)
	mux.HandleFunc("POST /frame", s.postFrame)
	mux.HandleFunc("GET /frame.jpg", s.getFrame)
	mux.HandleFunc("POST /snapshot", s.snapshot)
	return mux
}

func (s *webServer) getSettings(w http.ResponseWriter, r *http.Request) {
	b, err := os.ReadFile(s.settings)
	if err != nil {
		http.Error(w, "no settings file", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(b)
}

// Settings is the settings file: what the page applies on Connect and grab
// applies by default. Exposure and Gain are "auto" or a number; WB is
// "auto" or the red, green, blue gains (absent = auto).
type Settings struct {
	Size     int     `json:"size"`
	Quality  int     `json:"quality"`
	Exposure autoNum `json:"exposure"`
	Gain     autoNum `json:"gain"`
	WB       autoWB  `json:"wb"`
	Mirror   bool    `json:"mirror"`
	Flip     bool    `json:"flip"`
}

// autoWB is "auto" (Auto) or three gains.
type autoWB struct {
	Auto bool
	RGB  [3]float64
}

func (a autoWB) MarshalJSON() ([]byte, error) {
	if a.Auto {
		return []byte(`"auto"`), nil
	}
	return json.Marshal(a.RGB)
}

func (a *autoWB) UnmarshalJSON(b []byte) error {
	if string(b) == `"auto"` {
		*a = autoWB{Auto: true}
		return nil
	}
	var g []float64
	if err := json.Unmarshal(b, &g); err != nil {
		return err
	}
	if len(g) != 3 {
		return errors.New(`wb must be "auto" or [r, g, b]`)
	}
	*a = autoWB{RGB: [3]float64(g)}
	return nil
}

// autoNum is "auto" (Auto) or a number.
type autoNum struct {
	Auto bool
	V    float64
}

func (a autoNum) MarshalJSON() ([]byte, error) {
	if a.Auto {
		return []byte(`"auto"`), nil
	}
	return json.Marshal(a.V)
}

func (a *autoNum) UnmarshalJSON(b []byte) error {
	if string(b) == `"auto"` {
		*a = autoNum{Auto: true}
		return nil
	}
	a.Auto = false
	return json.Unmarshal(b, &a.V)
}

func (st Settings) validate() error {
	switch {
	case st.Size < 0 || st.Size > 4:
		return errors.New("size must be 0..4")
	case st.Quality < 1 || st.Quality > 63:
		return errors.New("quality must be 1..63")
	case !st.Exposure.Auto && (st.Exposure.V < 1 || st.Exposure.V > 3_000_000):
		return errors.New(`exposure must be "auto" or 1..3000000 µs`)
	case !st.Gain.Auto && (st.Gain.V < 1 || st.Gain.V > 15.5):
		return errors.New(`gain must be "auto" or 1..15.5`)
	}
	for _, g := range st.WB.RGB {
		if !st.WB.Auto && (g < 0 || g > 3.99) {
			return errors.New(`wb gains must be 0..3.99`)
		}
	}
	return nil
}

// decodeSettings reads settings JSON; a missing "wb" means auto.
func decodeSettings(r io.Reader) (Settings, error) {
	st := Settings{WB: autoWB{Auto: true}}
	dec := json.NewDecoder(io.LimitReader(r, 4096))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&st); err != nil {
		return st, err
	}
	if dec.More() {
		return st, errors.New("trailing data after the settings object")
	}
	return st, st.validate()
}

// putSettings validates the posted settings and writes the settings file
// (complete JSON, all fields), replacing it atomically.
func (s *webServer) putSettings(w http.ResponseWriter, r *http.Request) {
	st, err := decodeSettings(r.Body)
	if err != nil {
		http.Error(w, "settings: "+err.Error(), http.StatusBadRequest)
		return
	}
	b, _ := json.MarshalIndent(st, "", "  ") // cannot fail: plain values
	tmp := s.settings + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := os.Rename(tmp, s.settings); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"path": s.settings})
}

func (s *webServer) postFrame(w http.ResponseWriter, r *http.Request) {
	b, err := io.ReadAll(io.LimitReader(r.Body, maxFrame+1))
	if err != nil || len(b) < 4 || len(b) > maxFrame || b[0] != 0xFF || b[1] != 0xD8 || b[len(b)-2] != 0xFF || b[len(b)-1] != 0xD9 {
		http.Error(w, "not a JPEG up to 1 MiB", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.jpeg, s.at = b, time.Now()
	s.seq++
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

// latest returns the last posted image if it is fresh.
func (s *webServer) latest(now time.Time) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.jpeg == nil || now.Sub(s.at) > staleAfter {
		return nil, errors.New("no live image: the client page is not connected (use grab)")
	}
	return s.jpeg, nil
}

func (s *webServer) getFrame(w http.ResponseWriter, r *http.Request) {
	j, err := s.latest(time.Now())
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(j)
}

func (s *webServer) snapshot(w http.ResponseWriter, r *http.Request) {
	j, err := s.latest(time.Now())
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	path, err := saveSnapshot(s.out, s.keep, j, time.Now())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, "{\"path\": %q}\n", path)
}
