package retrieve

import (
	"fmt"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// BenchmarkFind runs a mixed query over ~2k synthetic linked documents.
func BenchmarkFind(b *testing.B) {
	vocab := []string{"client", "permission", "subject", "credential", "lease", "grant", "authority", "nats",
		"motor", "odometry", "telemetry", "imu", "calibration", "limiter", "gateway", "cache", "token", "raft"}
	for _, o := range openers {
		b.Run(o.name, func(b *testing.B) {
			s := o.open(b)
			b.Cleanup(func() { _ = s.Close() })
			const n = 2000
			for i := range n {
				var secs []sec
				for j := range 4 {
					text := fmt.Sprintf("%s %s %s %s", vocab[(i+j)%len(vocab)], vocab[(i*7+j)%len(vocab)], vocab[(i*13+j*3)%len(vocab)], filler(20+j))
					secs = append(secs, sec{heading: vocab[(i+j*5)%len(vocab)], text: text})
				}
				tgt := (i*31 + 1) % n
				d := doc(fmt.Sprintf("area%02d/doc%04d.md", i%50, i), secs...).links(fmt.Sprintf("area%02d/doc%04d.md", tgt%50, tgt))
				if i%10 == 0 {
					d.definesID(fmt.Sprintf("DOC-%d", i))
				} else {
					d.mentionsID(fmt.Sprintf("DOC-%d", i/10*10))
				}
				load(b, s, d)
			}
			r := New(s)
			q := knowledge.Query{Text: "how is a client permission lease granted DOC-120"}
			find(b, r, q) // warm the docs cache
			b.ResetTimer()
			for b.Loop() {
				find(b, r, q)
			}
		})
	}
}
