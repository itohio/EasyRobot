//go:build unitv

// Command camid powers the UnitV camera up, reads its sensor id over SCCB and
// prints it with the clock rates once a second (milestone M3).
package main

import (
	"machine"
	"strconv"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera"
	"github.com/itohio/EasyRobot/drivers/unitv"
)

func main() {
	uart := machine.UART0
	uart.Configure(machine.UARTConfig{BaudRate: 115200})
	var line []byte
	hex := func(b uint8) {
		line = append(line, "0x"...)
		if b < 0x10 {
			line = append(line, '0')
		}
		line = strconv.AppendUint(line, uint64(b), 16)
	}
	next := time.Now()
	for {
		id, err := camera.Identify(camera.Config{}, unitv.CameraPins)
		c := camera.ClockInfo()
		line = append(line[:0], "camid pll0="...)
		line = strconv.AppendUint(line, uint64(c.PLL0), 10)
		line = append(line, " aclk="...)
		line = strconv.AppendUint(line, uint64(c.ACLK), 10)
		line = append(line, " apb1="...)
		line = strconv.AppendUint(line, uint64(c.APB1), 10)
		line = append(line, " xclk="...)
		line = strconv.AppendUint(line, uint64(c.XCLK), 10)
		line = append(line, " sccb="...)
		line = strconv.AppendUint(line, uint64(c.SCCB), 10)
		if err != nil {
			line = append(line, " err="...)
			line = append(line, err.Error()...)
		} else {
			line = append(line, " sensor="...)
			line = append(line, id.Model.String()...)
			line = append(line, " addr="...)
			hex(id.Addr)
			line = append(line, " pid="...)
			hex(id.PID)
			line = append(line, " ver="...)
			hex(id.VER)
		}
		line = append(line, '\r', '\n')
		uart.Write(line)
		next = next.Add(time.Second)
		for d := time.Until(next); d > 0; d = time.Until(next) {
			time.Sleep(d)
		}
	}
}
