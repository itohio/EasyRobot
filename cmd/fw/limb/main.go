// +build sam,xiao

package main

//go:generate tinygo flash -target=xiao

import (
	"machine"
	"math/rand"
	"time"

	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
)

func blink(led machine.Pin, t time.Duration) {
	for {
		time.Sleep(t)
		led.Set(!led.Get())
	}
}

func main() {
	led := machine.LED
	led.Configure(machine.PinConfig{Mode: machine.PinOutput})

	cfg := []servos.Motor{
		servos.NewDefaultConfig(uint32(machine.D8)),
		servos.NewDefaultConfig(uint32(machine.D9)),
		servos.NewDefaultConfig(uint32(machine.D10)),
	}

	leg, err := servos.New(cfg)
	if err != nil {
		blink(led, time.Millisecond*1500)
	}

	pos := []float32{0, 0, 0}
	for {
		for i := range pos {
			pos[i] = float32(90 - rand.Intn(180))
		}
		leg.Set(pos)

		time.Sleep(time.Second)
		led.Set(!led.Get())
	}
}
