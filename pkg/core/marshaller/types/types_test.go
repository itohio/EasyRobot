package types

import (
	"context"
	"testing"
)

func TestWithContextSetsContext(t *testing.T) {
	ctx := context.WithValue(context.Background(), "key", "value")

	opts := Options{}
	option := WithContext(ctx)
	option.Apply(&opts)

	if opts.Context == nil {
		t.Fatalf("expected context to be set, got nil")
	}
	if opts.Context != ctx {
		t.Fatalf("expected context %v, got %v", ctx, opts.Context)
	}
}

func TestWithContextNilDefaultsToBackground(t *testing.T) {
	option := WithContext(nil)

	opts := Options{}
	option.Apply(&opts)

	if opts.Context == nil {
		t.Fatalf("expected non-nil context")
	}
	if opts.Context != context.Background() {
		t.Fatalf("expected background context, got %v", opts.Context)
	}
}

func TestWithContextNilOptionsDoesNotPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected no panic, got %v", r)
		}
	}()

	option := WithContext(context.Background())
	option.Apply(nil)
}

func TestFrameStreamClose(t *testing.T) {
	closed := false
	stream := FrameStream{
		C: make(<-chan Frame),
		close: func() {
			closed = true
		},
	}

	stream.Close()

	if !closed {
		t.Fatalf("expected close callback to be invoked")
	}
}

func TestFrameStreamCloseNoCallback(t *testing.T) {
	stream := FrameStream{
		C: make(<-chan Frame),
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected no panic when close callback is nil, got %v", r)
		}
	}()

	stream.Close()
}
