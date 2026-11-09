package main

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

const (
	// resendIdle is how long a partly received image may stay silent before
	// the missing chunks are Nak'ed again.
	resendIdle = 200 * time.Millisecond
	// resendRounds bounds the Nak rounds per image; then the image is given
	// up and Ack'ed so the sender moves on.
	resendRounds = 10
)

// resendStats are the counters of a resend session.
type resendStats struct {
	images, saved, naks, gaveUp int
	abandoned, lost             uint64
	firstTS, lastTS             uint32
}

func (s resendStats) fps() float64 {
	if s.images < 2 || s.lastTS == s.firstTS {
		return 0
	}
	return float64(s.images-1) / (float64(s.lastTS-s.firstTS) / 1e6)
}

// runResend receives with chunk resend (frame.Control): every CRC-valid
// chunk goes into an Assembler; when an image's last chunk arrives, or the
// image stays silent for resendIdle, each missing chunk is Nak'ed; a complete
// image is Ack'ed and saved. fps comes from the device timestamps.
func runResend(ctx context.Context, c config, dec *frame.Decoder, port io.Writer, stdout io.Writer) (resendStats, error) {
	type item struct {
		c   frame.Chunk
		err error
	}
	ch := make(chan item, 64)
	go func() {
		for {
			ck, err := dec.NextChunk()
			select {
			case ch <- item{ck, err}:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()

	var (
		st      resendStats
		asm     frame.Assembler
		rounds  int
		seqSeen bool
		lastSeq uint32
		b       [frame.ControlSize]byte
		miss    []uint16
	)
	send := func(k frame.ControlKind, seq uint32, chunk uint16) error {
		frame.Control{Kind: k, Seq: seq, Chunk: chunk}.Put(&b)
		_, err := port.Write(b[:])
		return err
	}
	nakMissing := func() error {
		seq, _, ok := asm.Current()
		if !ok {
			return nil
		}
		miss = asm.Missing(miss[:0])
		for _, i := range miss {
			if err := send(frame.Nak, seq, i); err != nil {
				return err
			}
			st.naks++
		}
		rounds++
		return nil
	}
	idle := time.NewTimer(resendIdle)
	defer idle.Stop()

	for c.n == 0 || st.saved < c.n {
		select {
		case <-ctx.Done():
			st.abandoned = asm.Abandon
			return st, ctx.Err()
		case it := <-ch:
			if it.err != nil {
				st.abandoned = asm.Abandon
				return st, it.err
			}
			idle.Reset(resendIdle)
			if seqSeen && it.c.Seq != lastSeq {
				if gap := it.c.Seq - lastSeq; gap > 1 && gap < 1<<31 {
					st.lost += uint64(gap - 1)
				}
			}
			seqSeen, lastSeq = true, it.c.Seq
			img, done := asm.Add(it.c)
			if done {
				rounds = 0
				if err := send(frame.Ack, img.Seq, 0); err != nil {
					return st, err
				}
				if st.images == 0 {
					st.firstTS = img.TS
				}
				st.lastTS = img.TS
				st.images++
				if err := save(c.out, img, stdout); err != nil {
					fmt.Fprintf(stdout, "camframe: image %d: %v\n", img.Seq, err)
					continue
				}
				st.saved++
				continue
			}
			if it.c.ChunkIdx == it.c.ChunkCnt-1 {
				if err := nakMissing(); err != nil {
					return st, err
				}
			}
		case <-idle.C:
			if seq, _, ok := asm.Current(); ok {
				if rounds >= resendRounds {
					st.gaveUp++
					asm.Reset()
					rounds = 0
					if err := send(frame.Ack, seq, 0); err != nil {
						return st, err
					}
				} else if err := nakMissing(); err != nil {
					return st, err
				}
			}
			idle.Reset(resendIdle)
		}
	}
	st.abandoned = asm.Abandon
	return st, nil
}
