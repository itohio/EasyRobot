package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
	"time"
)

// pollReader mimics the serial port: data in small pieces, 0, nil between
// them, then an error.
type pollReader struct {
	parts [][]byte
	end   error
}

func (r *pollReader) Read(p []byte) (int, error) {
	if len(r.parts) == 0 {
		return 0, r.end
	}
	part := r.parts[0]
	r.parts = r.parts[1:]
	return copy(p, part), nil // an empty part is the 0, nil poll timeout
}

func TestAsyncReaderDeliversEverythingThenTheError(t *testing.T) {
	var want []byte
	src := &pollReader{end: io.ErrUnexpectedEOF}
	for i := range 500 {
		part := bytes.Repeat([]byte{byte(i)}, i%97+1)
		want = append(want, part...)
		src.parts = append(src.parts, part, nil)
	}
	a := newAsyncReader(context.Background(), src)
	var got []byte
	buf := make([]byte, 333) // reads smaller than the goroutine's blocks
	for {
		n, err := a.Read(buf)
		got = append(got, buf[:n]...)
		if err != nil {
			if !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("err = %v, want the source's error", err)
			}
			break
		}
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("got %d bytes, want %d (content differs: %v)", len(got), len(want), len(got) == len(want))
	}
}

// blockReader never returns data, like a quiet serial port.
type blockReader struct{ stop chan struct{} }

func (r blockReader) Read(p []byte) (int, error) { <-r.stop; return 0, io.EOF }

func TestAsyncReaderHonoursContextWhileWaiting(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	a := newAsyncReader(ctx, blockReader{stop})
	start := time.Now()
	_, err := a.Read(make([]byte, 16))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("Read returned after %v, not at the deadline", d)
	}
}
