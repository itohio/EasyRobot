//go:build !darwin && !linux

package main

import "errors"

// port is unavailable on this OS.
type port struct{}

var errNoSerial = errors.New("camclient: serial ports are supported on darwin and linux only")

func openPort(string) (*port, error)    { return nil, errNoSerial }
func findPort(string) (string, error)   { return "", errNoSerial }
func (*port) Read([]byte) (int, error)  { return 0, errNoSerial }
func (*port) Write([]byte) (int, error) { return 0, errNoSerial }
func (*port) Close() error              { return nil }
