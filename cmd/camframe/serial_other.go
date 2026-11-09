//go:build !darwin && !linux

package main

import "errors"

// port is unavailable on this OS; -in file mode still works.
type port struct{}

func openPort(string, int) (*port, error) {
	return nil, errors.New("serial ports are supported on darwin and linux only")
}

func (*port) Read([]byte) (int, error)  { return 0, errors.New("no serial port") }
func (*port) Write([]byte) (int, error) { return 0, errors.New("no serial port") }
func (*port) Close() error              { return nil }
