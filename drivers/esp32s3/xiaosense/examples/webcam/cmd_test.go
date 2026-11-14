package main

import "testing"

func TestParseCommand(t *testing.T) {
	for _, tt := range []struct {
		line string
		want command
	}{
		{"", command{kind: cmdNone}},
		{"G", command{kind: cmdGrab, quality: defaultQuality}},
		{"G 2 40", command{kind: cmdGrab, size: 2, quality: 40}},
		{"G 0 10", command{kind: cmdGrab, quality: 10}}, // full sensor range (owner); the page warns below 25
		{"G 0 0", command{kind: cmdGrab, quality: 1}},
		{"G 0 99", command{kind: cmdGrab, quality: maxQuality}},
		{"G 5", command{kind: cmdBad}}, // no such size
		{"S", command{kind: cmdStart}},
		{"X", command{kind: cmdStop}},
		{"?", command{kind: cmdStatus}},
		{"Z 4", command{kind: cmdSize, size: 4}},
		{"Z", command{kind: cmdBad}},
		{"Q 25", command{kind: cmdQuality, quality: 25}},
		{"Q 24", command{kind: cmdQuality, quality: 24}},
		{"Q 1", command{kind: cmdQuality, quality: 1}},
		{"E 500000", command{kind: cmdExposure, exposure: 500_000}},
		{"E 9000000", command{kind: cmdExposure, exposure: maxExposureUS}},
		{"E 0", command{kind: cmdExposure, exposure: 1}},
		{"E a", command{kind: cmdExposure, auto: true}},
		{"E -5", command{kind: cmdBad}},
		{"I 1", command{kind: cmdGain, gain: 1}},
		{"I 0.5", command{kind: cmdGain, gain: minGain}},
		{"I 40", command{kind: cmdGain, gain: maxGain}},
		{"I NaN", command{kind: cmdBad}},
		{"I a", command{kind: cmdGain, auto: true}},
		{"M 1 0", command{kind: cmdMirror, h: true}},
		{"M 0 1", command{kind: cmdMirror, v: true}},
		{"M 1", command{kind: cmdBad}},
		{"M 2 0", command{kind: cmdBad}},
		{"hello", command{kind: cmdBad}},
		{"  Z\t3 ", command{kind: cmdSize, size: 3}},
	} {
		if got := parseCommand([]byte(tt.line)); got != tt.want {
			t.Errorf("%q: %+v, want %+v", tt.line, got, tt.want)
		}
	}
}
