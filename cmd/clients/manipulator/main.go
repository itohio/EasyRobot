package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
	"go.bug.st/serial.v1"
)

func main() {
	help := flag.Bool("help", false, "Help")
	listPorts := flag.Bool("list", false, "List all ports")
	port := flag.String("port", "COM1", "COM port to the device")
	baud := flag.Int("baud", 115200, "COM port baud rate")

	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	if *listPorts {
		list, err := serial.GetPortsList()
		if err != nil {
			panic(err)
		}

		fmt.Println("Available serial ports:")
		for i, p := range list {
			fmt.Println(i, "\t", p)
		}
		return
	}

	mode := &serial.Mode{
		BaudRate: *baud,
	}

	rw, err := serial.Open(*port, mode)
	if err != nil {
		panic(err)
	}
	defer rw.Close()

	go func() {
		for {
			data := [1]byte{}
			n, _ := rw.Read(data[:])
			if n > 0 {
				fmt.Print(string(data[:]))
			}
		}
	}()

	manipulator := servos.NewClient(rw)

	manipulator.Configure([]servos.Motor{
		servos.NewDefaultConfig(servos.WithMicroseconds(500, 2500, 1500, 180)),
		servos.NewDefaultConfig(servos.WithMicroseconds(500, 2500, 1500, 180)),
		servos.NewDefaultConfig(servos.WithMicroseconds(500, 2500, 1500, 180)),
	},
	)

	state := vec.New(3)
	state[0] = -45
	state[1] = 0
	state[2] = 45
	for {
		fmt.Println(">>> Set state ", state)
		if err := manipulator.Set(state); err != nil {
			panic(err)
		}
		time.Sleep(time.Second)

		for i := range state {
			state[i] = float32(90 - 45*rand.Intn(4))
		}
	}
}
