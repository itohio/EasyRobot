package main

import (
	"bufio"
	"bytes"
	"strings"
)

// calloutFor finds, in ioreg -l output, the first IOCalloutDevice after the
// device whose "USB Serial Number" is sn and before the next device's serial.
func calloutFor(ioreg []byte, sn string) string {
	want := `"USB Serial Number" = "` + sn + `"`
	in := false
	sc := bufio.NewScanner(bytes.NewReader(ioreg))
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		l := sc.Text()
		switch {
		case strings.Contains(l, `"USB Serial Number" = `):
			in = strings.Contains(l, want)
		case in && strings.Contains(l, `"IOCalloutDevice" = "`):
			v := l[strings.Index(l, `"IOCalloutDevice" = "`)+len(`"IOCalloutDevice" = "`):]
			return strings.TrimSuffix(strings.TrimSpace(v), `"`)
		}
	}
	return ""
}
