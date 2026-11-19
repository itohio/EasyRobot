//go:build !k210

package kpu

func open(bool) (engine, error) { return nil, ErrNoKPU }

func lendable() ([]byte, error) { return nil, ErrNoKPU }
