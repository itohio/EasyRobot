package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/actuator"
	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
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

	manipulator := servos.New(rw)

	manipulator.Configure(
		actuator.WithServoConfig([]servos.Motor{
			servos.NewMotorConfig(servos.WithMicroseconds(500, 2500, 1500, 180)),
			servos.NewMotorConfig(servos.WithMicroseconds(500, 2500, 1500, 180)),
			servos.NewMotorConfig(servos.WithMicroseconds(500, 2500, 1500, 180)),
		}),
		actuator.WithMotionConfig([]kinematics.Motion{
			{Velocity: 50, Acceleration: 10, Jerk: 1},
			{Velocity: 50, Acceleration: 10, Jerk: 1},
			{Velocity: 50, Acceleration: 10, Jerk: 1},
		}),
		actuator.WithPlanarKinematics([]kinematics.PlanarJoint{
			{MinAngle: -90, MaxAngle: 90, Length: 39},
			{MinAngle: -90, MaxAngle: 90, Length: 45},
			{MinAngle: -10, MaxAngle: 90, Length: 100},
		}),
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
		time.Sleep(time.Second * 3)

		for i := range state {
			state[i] = float32(90 - 45*rand.Intn(4))
		}
	}
}
