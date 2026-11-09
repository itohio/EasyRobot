package main

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

const (
	// resendRounds bounds the Nak rounds per image; then the image is given
	// up and Ack'ed so the sender moves on.
	resendRounds = 10
	// minResendIdle is the floor of the idle time (fast links).
	minResendIdle = 200 * time.Millisecond
)

// resendIdle is how long a partly received image may stay silent before
// its missing chunks are Nak'ed again: three airtimes of a full chunk at
// baud (8N1 = 10 bits per byte) plus 100 ms, at least minResendIdle. A
// fixed 200 ms fired in the middle of a chunk below ~200 kbaud and gave
// clean images up (review R-ER-K210-camera M2).
func resendIdle(baud int) time.Duration {
	if baud <= 0 {
		return minResendIdle
	}
	air := time.Duration(int64(frame.HeaderSize+frame.MaxPayload) * 10 * int64(time.Second) / int64(baud))
	return max(3*air+100*time.Millisecond, minResendIdle)
}

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
// image stays silent for resendIdle(c.baud), each missing chunk is Nak'ed; a
// complete image is Ack'ed and saved. fps comes from the device timestamps.
//
// dec must read through a context that stopDec cancels: runResend cancels
// it and waits for its reader goroutine before returning, so the caller can
// use dec.Stats and dec.Text without a race (review M1).
func runResend(ctx context.Context, c config, dec *frame.Decoder, stopDec context.CancelFunc, port io.Writer, stdout io.Writer) (resendStats, error) {
	type item struct {
		c   frame.Chunk
		err error
	}
	ch := make(chan item, 64)
	done := make(chan struct{})
	quit := make(chan struct{})
	go func() {
		defer close(done)
		for {
			ck, err := dec.NextChunk()
			select {
			case ch <- item{ck, err}:
			case <-quit:
				return
			}
			if err != nil {
				return
			}
		}
	}()
	defer func() {
		stopDec()
		close(quit)
		<-done
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
	idleFor := resendIdle(c.baud)
	idle := time.NewTimer(idleFor)
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
			idle.Reset(idleFor)
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
			idle.Reset(idleFor)
		}
	}
	st.abandoned = asm.Abandon
	return st, nil
}
