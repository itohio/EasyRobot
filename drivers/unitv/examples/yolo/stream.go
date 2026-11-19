package main

import (
	"io"
	"runtime"
	"strconv"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

// streamer writes each camera frame as a frame v1 image, then its detection
// record (drivers/camera/frame), then log lines stating what the record
// holds, plus heap figures. Buffers are allocated once; send allocates
// nothing.
type streamer struct {
	w    io.Writer
	enc  *frame.Encoder
	meta frame.Meta
	rec  frame.Detections
	dets [frame.MaxDetections]frame.Detection
	buf  []byte
	ms   runtime.MemStats
}

func newStreamer(w io.Writer, meta frame.Meta) *streamer {
	return &streamer{w: w, enc: frame.NewEncoder(w), meta: meta,
		buf: make([]byte, 0, frame.DetHeaderSize+frame.DetSize*frame.MaxDetections+4+4096)}
}

// send writes img (ts: capture start, device µs) as one image, the record
// of dets (image pixels; at most frame.MaxDetections are sent) with the
// stage times t, and the log lines naming classes from labels. fnv holds
// the detector's Check hashes, logged so a host run of the same image can
// be compared bit for bit.
func (s *streamer) send(img []byte, ts uint32, dets []frame.Detection, labels []string, t frame.StageTimes, fnv [3]uint32) error {
	n := copy(s.dets[:], dets)
	s.rec = frame.Detections{Seq: s.enc.Seq(), Width: s.meta.Width, Height: s.meta.Height, Times: t, Dets: s.dets[:n]}
	if err := s.enc.WriteImage(s.meta, ts, img); err != nil {
		return err
	}
	s.buf = frame.AppendDetections(s.buf[:0], &s.rec)
	if _, err := s.w.Write(s.buf); err != nil {
		return err
	}
	runtime.ReadMemStats(&s.ms)
	s.buf = appendRecordLog(s.buf[:0], &s.rec, labels)
	s.buf = append(s.buf, "yolo: check input_fnv "...)
	s.buf = strconv.AppendUint(s.buf, uint64(fnv[0]), 16)
	s.buf = append(s.buf, " out0_fnv "...)
	s.buf = strconv.AppendUint(s.buf, uint64(fnv[1]), 16)
	s.buf = append(s.buf, " out1_fnv "...)
	s.buf = strconv.AppendUint(s.buf, uint64(fnv[2]), 16)
	s.buf = append(s.buf, " total_alloc "...)
	s.buf = strconv.AppendUint(s.buf, s.ms.TotalAlloc, 10)
	s.buf = append(s.buf, " mallocs "...)
	s.buf = strconv.AppendUint(s.buf, s.ms.Mallocs, 10)
	s.buf = append(s.buf, " heap_inuse "...)
	s.buf = strconv.AppendUint(s.buf, s.ms.HeapInuse, 10)
	s.buf = append(s.buf, '\r', '\n')
	_, err := s.w.Write(s.buf)
	return err
}

// appendRecordLog appends the log lines of a detection record: one frame
// line (sequence, count, stage times in ms) and one line per detection
// (sequence, class name from labels, score, box in image pixels).
func appendRecordLog(b []byte, d *frame.Detections, labels []string) []byte {
	ms := func(b []byte, us uint32) []byte { return strconv.AppendFloat(b, float64(us)/1000, 'f', 1, 64) }
	b = append(b, "yolo: frame "...)
	b = strconv.AppendUint(b, uint64(d.Seq), 10)
	b = append(b, " dets "...)
	b = strconv.AppendInt(b, int64(len(d.Dets)), 10)
	b = append(b, " capture_ms "...)
	b = ms(b, d.Times.Capture)
	b = append(b, " letterbox_ms "...)
	b = ms(b, d.Times.Letterbox)
	b = append(b, " detect_ms "...)
	b = ms(b, d.Times.Detect)
	b = append(b, " send_prev_ms "...)
	b = ms(b, d.Times.Send)
	b = append(b, '\r', '\n')
	for _, x := range d.Dets {
		b = append(b, "yolo: det "...)
		b = strconv.AppendUint(b, uint64(d.Seq), 10)
		b = append(b, ' ')
		if int(x.Class) < len(labels) {
			b = append(b, labels[x.Class]...)
		} else {
			b = strconv.AppendUint(append(b, "class"...), uint64(x.Class), 10)
		}
		b = append(b, ' ')
		b = strconv.AppendFloat(b, float64(x.Score)/65535, 'f', 4, 64)
		for _, v := range [4]int16{x.X0, x.Y0, x.X1, x.Y1} {
			b = append(b, ' ')
			b = strconv.AppendInt(b, int64(v), 10)
		}
		b = append(b, '\r', '\n')
	}
	return b
}
