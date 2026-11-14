package main

import "strconv"

// The webcam's text command set (one line each, host → board):
//
//	G [size] [quality]  one image (defaults: size 0, quality 30)
//	S                   start streaming images back to back
//	X                   stop streaming
//	Z <size>            frame size: 0 = 2048×1536, 1 = 1600×1200, 2 = 1024×768, 3 = 800×600, 4 = 640×480
//	Q <quality>         JPEG quantisation 1 … 63 (smaller = finer, larger JPEG)
//	E <µs> | E a        manual exposure time (up to 3,000,000) | auto exposure
//	I <gain> | I a      manual analogue gain (1 … 15.5) | auto gain
//	M <h> <v>           mirror left-right (h) and flip top-bottom (v), 0 or 1
//	W a | W <r> <g> <b> auto white balance | manual red, green, blue gains (0 … 3.99)
//	?                   status line
//
// The whole quality range is accepted (owner). Below 25, noisy frames
// (high gain, dark) can come out corrupted (ER-ESP32CAM-E7 evidence): the
// status line counts failed captures (fails=) so the host can warn. Every
// command except G and S answers with a status line.

// Sizes the webcam offers, indexed by the Z and G size argument.
var sizeW = [...]uint16{2048, 1600, 1024, 800, 640}
var sizeH = [...]uint16{1536, 1200, 768, 600, 480}

const (
	minQuality     = 1
	maxQuality     = 63
	defaultQuality = 30
	maxExposureUS  = 3_000_000
	minGain        = 1
	maxGain        = 15.5
	maxWB          = 3.99 // the 12-bit gain registers hold up to 4095/1024
)

type cmdKind uint8

const (
	cmdNone cmdKind = iota
	cmdGrab
	cmdStart
	cmdStop
	cmdSize
	cmdQuality
	cmdExposure
	cmdGain
	cmdMirror
	cmdWB
	cmdStatus
	cmdBad
)

// command is one parsed request. Auto is set for "E a" and "I a".
type command struct {
	kind     cmdKind
	size     int
	quality  uint8
	exposure uint32
	gain     float32
	auto     bool
	h, v     bool       // M
	wb       [3]float32 // W: red, green, blue gains
}

// parseCommand parses one line. Unknown or malformed lines give cmdBad, an
// empty line cmdNone. Numbers out of range are clamped to the range; numbers
// that don't fit their type at all (e.g. E 4294967296) are malformed.
func parseCommand(line []byte) command {
	f, n := fields(line)
	if n == 0 {
		return command{kind: cmdNone}
	}
	arg := func(i int) (string, bool) {
		if i < n {
			return f[i], true
		}
		return "", false
	}
	switch f[0] {
	case "G":
		c := command{kind: cmdGrab, quality: defaultQuality}
		if a, ok := arg(1); ok {
			s, ok := parseSize(a)
			if !ok {
				return command{kind: cmdBad}
			}
			c.size = s
		}
		if a, ok := arg(2); ok {
			q, ok := parseQuality(a)
			if !ok {
				return command{kind: cmdBad}
			}
			c.quality = q
		}
		return c
	case "S":
		return command{kind: cmdStart}
	case "X":
		return command{kind: cmdStop}
	case "?":
		return command{kind: cmdStatus}
	case "Z":
		a, _ := arg(1)
		if s, ok := parseSize(a); ok {
			return command{kind: cmdSize, size: s}
		}
	case "Q":
		a, _ := arg(1)
		if q, ok := parseQuality(a); ok {
			return command{kind: cmdQuality, quality: q}
		}
	case "E":
		a, _ := arg(1)
		if a == "a" {
			return command{kind: cmdExposure, auto: true}
		}
		if v, err := strconv.ParseUint(a, 10, 32); err == nil {
			return command{kind: cmdExposure, exposure: uint32(min(max(v, 1), maxExposureUS))}
		}
	case "M":
		h, okH := arg(1)
		v, okV := arg(2)
		if okH && okV && (h == "0" || h == "1") && (v == "0" || v == "1") {
			return command{kind: cmdMirror, h: h == "1", v: v == "1"}
		}
	case "W":
		if a, _ := arg(1); a == "a" && n == 2 {
			return command{kind: cmdWB, auto: true}
		}
		if n != 4 {
			break
		}
		c := command{kind: cmdWB}
		for i := range c.wb {
			v, err := strconv.ParseFloat(f[i+1], 32)
			if err != nil || v != v {
				return command{kind: cmdBad}
			}
			c.wb[i] = float32(min(max(v, 0), maxWB))
		}
		return c
	case "I":
		a, _ := arg(1)
		if a == "a" {
			return command{kind: cmdGain, auto: true}
		}
		if v, err := strconv.ParseFloat(a, 32); err == nil && v == v {
			return command{kind: cmdGain, gain: float32(min(max(v, minGain), maxGain))}
		}
	}
	return command{kind: cmdBad}
}

func parseSize(a string) (int, bool) {
	v, err := strconv.Atoi(a)
	if err != nil || v < 0 || v >= len(sizeW) {
		return 0, false
	}
	return v, true
}

func parseQuality(a string) (uint8, bool) {
	v, err := strconv.Atoi(a)
	if err != nil || v < 0 {
		return 0, false
	}
	return uint8(min(max(v, minQuality), maxQuality)), true
}

// fields splits line at spaces into at most 4 fields, without allocating
// beyond the strings themselves.
func fields(line []byte) (f [4]string, n int) {
	start := -1
	for i := 0; i <= len(line); i++ {
		if i == len(line) || line[i] == ' ' || line[i] == '\t' {
			if start >= 0 && n < len(f) {
				f[n] = string(line[start:i])
				n++
			}
			start = -1
		} else if start < 0 {
			start = i
		}
	}
	return f, n
}
