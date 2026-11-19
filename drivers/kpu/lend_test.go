package kpu

import (
	"errors"
	"testing"
)

var errPower = errors.New("power failed")

// get and start stand in for powering the KPU memory and creating a
// Runner; calls counts how often they ran.
type fakeKPU struct {
	calls    int
	getErr   error
	startErr error
	memory   []byte
}

func (f *fakeKPU) get() ([]byte, error) {
	f.calls++
	if f.getErr != nil {
		return nil, f.getErr
	}
	return f.memory, nil
}

func (f *fakeKPU) start() error {
	f.calls++
	return f.startErr
}

// Once the memory is lent, no Runner can start (and the KPU isn't touched).
func TestLendThenUseIsBusy(t *testing.T) {
	var o memOwner
	f := &fakeKPU{memory: make([]byte, 16)}
	b, err := o.lend(f.get)
	if err != nil || len(b) != 16 {
		t.Fatalf("lend: %d bytes, %v", len(b), err)
	}
	if err := o.use(f.start); !errors.Is(err, ErrBusy) {
		t.Errorf("use after lend: %v, want ErrBusy", err)
	}
	if _, err := o.lend(f.get); !errors.Is(err, ErrBusy) {
		t.Errorf("second lend: %v, want ErrBusy", err)
	}
	if f.calls != 1 {
		t.Errorf("KPU touched %d times, want 1 (the first lend)", f.calls)
	}
}

// Once a Runner exists, the memory can't be lent.
func TestUseThenLendIsBusy(t *testing.T) {
	var o memOwner
	f := &fakeKPU{}
	if err := o.use(f.start); err != nil {
		t.Fatal(err)
	}
	if _, err := o.lend(f.get); !errors.Is(err, ErrBusy) {
		t.Errorf("lend after use: %v, want ErrBusy", err)
	}
	if err := o.use(f.start); err != nil {
		t.Errorf("second Runner: %v", err)
	}
}

// A failed lend or a failed Runner start claims nothing.
func TestFailuresClaimNothing(t *testing.T) {
	var o memOwner
	if _, err := o.lend((&fakeKPU{getErr: errPower}).get); !errors.Is(err, errPower) {
		t.Fatalf("failing lend: %v", err)
	}
	if err := o.use((&fakeKPU{startErr: errPower}).start); !errors.Is(err, errPower) {
		t.Fatalf("failing use: %v", err)
	}
	if _, err := o.lend((&fakeKPU{memory: make([]byte, 1)}).get); err != nil {
		t.Errorf("lend after the failures: %v", err)
	}
}

// Off the K210 LendMemory reports no KPU and leaves New's answer unchanged.
func TestLendMemoryOffK210(t *testing.T) {
	if _, err := LendMemory(); !errors.Is(err, ErrNoKPU) {
		t.Errorf("LendMemory on the host: %v", err)
	}
	if _, err := New(model(t), nil); !errors.Is(err, ErrNoKPU) {
		t.Errorf("New after a failed LendMemory: %v, want ErrNoKPU", err)
	}
}
