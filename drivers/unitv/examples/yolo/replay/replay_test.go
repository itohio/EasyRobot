package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"reflect"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

func committed(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../web/replay.bin")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The committed file is what the generator writes (regenerate with
// go run ./drivers/unitv/examples/yolo/replay after changing it).
func TestReplayFileIsGenerated(t *testing.T) {
	if !bytes.Equal(committed(t), build()) {
		t.Error("web/replay.bin differs from build(): regenerate it")
	}
}

// The committed file decodes to images 0 and 2 (frame 1 is dropped on its
// corrupted chunk) and to the three known detection records.
func TestReplayFileDecodes(t *testing.T) {
	var text bytes.Buffer
	dec := frame.NewDecoder(bytes.NewReader(committed(t)))
	dec.Text = &text
	var seqs []uint32
	for {
		img, err := dec.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if img.Meta.Width != w || img.Meta.Height != h || img.Meta.Format != frame.RGB565LE {
			t.Errorf("image %d: %+v", img.Seq, img.Meta)
		}
		seqs = append(seqs, img.Seq)
	}
	if st := dec.Stats(); !reflect.DeepEqual(seqs, []uint32{0, 2}) || st.CRCErrors != 1 || st.Dropped != 1 {
		t.Errorf("images %v, stats %+v; want images [0 2], 1 CRC error, 1 dropped", seqs, st)
	}
	var got []frame.Detections
	b := text.Bytes()
	for i := 0; i+4 <= len(b); i++ {
		if binary.LittleEndian.Uint32(b[i:]) != frame.DetMagic {
			continue
		}
		r, n, err := frame.ParseDetections(b[i:])
		if err != nil {
			t.Fatalf("record at text offset %d: %v", i, err)
		}
		got = append(got, r)
		i += n - 1
	}
	if !reflect.DeepEqual(got, records) {
		t.Errorf("records\n got %+v\nwant %+v", got, records)
	}
}
