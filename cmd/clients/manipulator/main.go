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

	// manipulator.Configure([]servos.Motor{
	// 	servos.NewDefaultConfig(0),
	// 	servos.NewDefaultConfig(0),
	// 	servos.NewDefaultConfig(0),
	// },
	// )

	state := vec.New(3)
	for {
		for i := range state {
			state[i] = float32(90 - rand.Intn(180))
		}
		fmt.Println(">>> Set state")
		if err := manipulator.Set(state); err != nil {
			panic(err)
		}
		time.Sleep(time.Second)
	}
}
