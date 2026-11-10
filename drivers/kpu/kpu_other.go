//go:build !k210

package kpu

func open(bool) (engine, error) { return nil, ErrNoKPU }
