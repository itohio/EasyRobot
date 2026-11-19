package kpu

import (
	"errors"
	"sync"
)

// ErrBusy is returned by LendMemory once a Runner has been created, and by
// New once the KPU memory has been lent.
var ErrBusy = errors.New("kpu: KPU memory is lent to the CPU or used by a Runner")

// LendMemory powers the KPU's clocks (PLL1 and the AI clock, as New does)
// and lends its 2 MiB memory to the CPU as ordinary RAM, through the cached
// alias at 0x8060_0000: for example as the activation arena of a CPU
// network while the KPU itself is unused. The memory holds whatever was
// there before.
//
// The loan lasts for the rest of the program: New then returns ErrBusy, and
// so does a second LendMemory. LendMemory returns ErrBusy once New has
// created a Runner (Runners have no end of life, so the memory could still
// be in use by one). It returns ErrNoKPU off the K210.
func LendMemory() ([]byte, error) { return owner.lend(lendable) }

// memOwner records who holds the KPU memory: the CPU (lent) or the KPU
// (a Runner was created).
type memOwner struct {
	mu         sync.Mutex
	lent, used bool
}

// owner is the KPU memory's owner record for this program.
var owner memOwner

// lend runs get (power the KPU, return its memory) unless a Runner exists
// or the memory is already lent, and marks it lent on success.
func (o *memOwner) lend(get func() ([]byte, error)) ([]byte, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.lent || o.used {
		return nil, ErrBusy
	}
	b, err := get()
	if err != nil {
		return nil, err
	}
	o.lent = true
	return b, nil
}

// use runs start (set up the KPU and create a Runner) unless the memory is
// lent, and marks it used by a Runner when start succeeds.
func (o *memOwner) use(start func() error) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.lent {
		return ErrBusy
	}
	if err := start(); err != nil {
		return err
	}
	o.used = true
	return nil
}
