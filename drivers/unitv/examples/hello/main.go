//go:build unitv

// Command hello prints a numbered line every second on UARTHS (USB serial,
// 115200 baud): the M1 bring-up program for the UnitV target.
//
// The loop sleeps until a deadline checked against time.Now: on TinyGo 0.43
// (K210) a bare `write; time.Sleep(time.Second)` loop returns from every
// second Sleep at once (docs/evidence/ER-K210-M1.md).
package main

import (
	"machine"
	"strconv"
	"time"
)

func main() {
	uart := machine.UART0
	uart.Configure(machine.UARTConfig{BaudRate: 115200})
	var line []byte
	next := time.Now()
	for i := 0; ; i++ {
		line = append(line[:0], "unitv hello "...)
		line = strconv.AppendInt(line, int64(i), 10)
		line = append(line, '\r', '\n')
		uart.Write(line)
		next = next.Add(time.Second)
		for d := time.Until(next); d > 0; d = time.Until(next) {
			time.Sleep(d)
		}
	}
}
