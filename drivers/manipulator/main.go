//go:build sam && xiao

// NOTE: This package is designed for building only on TinyGo for the Seeed Studio XIAO series microcontroller.
// See https://tinygo.org/ for details on building and flashing.
// Usage:
//   - Raw mode: tinygo flash -target=xiao -tags logless .
//   - Planar mode: tinygo flash -target=xiao -tags logless,planar .
//   - DH mode: tinygo flash -target=xiao -tags logless,dh .

package main

//go:generate tinygo flash -target=xiao -tags logless

import (
	"context"
	"io"
	"machine"
	"sync"
	"time"

	"github.com/itohio/dndm"
	"github.com/itohio/dndm/endpoint/remote"
	"github.com/itohio/dndm/network"
	"github.com/itohio/dndm/network/serial"
	"github.com/itohio/dndm/network/stream"
	"github.com/itohio/dndm/x/bus"

	types "github.com/itohio/EasyRobot/types/control"
	types_math "github.com/itohio/EasyRobot/types/math"
	"github.com/itohio/EasyRobot/x/devices/servo"
	"github.com/itohio/EasyRobot/x/devices/xiao"
	vaj "github.com/itohio/EasyRobot/x/math/control/motion"
)

var (
	manipulatorConfig = []servo.Motor{
		servo.NewMotorConfig(servo.WithPin(uint32(machine.D8))),
		servo.NewMotorConfig(servo.WithPin(uint32(machine.D9))),
		servo.NewMotorConfig(servo.WithPin(uint32(machine.D10))),
	}
	motion = []vaj.VAJ1D{
		{},
		{},
		{},
	}

	manipulator *servo.ServoArray
	motionLock  sync.Mutex
	kinematics  kinematicsInterface // Set in handleKinematicsConfig for planar/dh modes
)

// getIntentPath returns the DNDM intent path based on build tags
func getIntentPath() string {
	// Build tags determine the mode:
	// - planar tag: planar mode
	// - dh tag: DH mode
	// - neither: raw mode
	// This is evaluated at compile time, so each build will have the correct path
	return getIntentPathImpl()
}

// kinematicsInterface is implemented by planar and dh kinematics
type kinematicsInterface interface {
	// Inverse calculates joint angles from end effector position
	// Returns joint angles in radians
	Inverse(targetX, targetY, targetZ float32, orientation *types_math.Quaternion) ([]float32, error)
	// Forward calculates end effector position from joint angles (for validation)
	Forward(jointAngles []float32) (x, y, z float32, err error)
}

func blink(led machine.Pin, t time.Duration) {
	for {
		time.Sleep(t)
		led.Set(!led.Get())
	}
}

func main() {
	led := machine.LED
	led.Configure(machine.PinConfig{Mode: machine.PinOutput})
	uart.Configure(machine.UARTConfig{TX: tx, RX: rx})

	defer blink(led, time.Millisecond*1500)

	// Create PWM device for servos
	pwm := xiao.NewPWMDevice()

	// Create servo array for array of servos
	var err error
	manipulator, err = servo.NewServoArray(pwm, manipulatorConfig)
	if err != nil {
		println("Failed to create servo array:", err.Error())
		return
	}

	ctx := context.Background()

	// Setup DNDM router with serial endpoint
	router, err := setupRouter(ctx)
	if err != nil {
		println("Failed to setup router:", err.Error())
		return
	}
	defer router.Close()

	// Declare intent based on build mode
	intentPath := getIntentPath()
	println("Manipulator mode:", intentPath)

	// Declare intent
	stateProducer, err := bus.NewProducer[*types.ManipulatorState](ctx, router, intentPath)
	if err != nil {
		println("Failed to create state producer:", err.Error())
		return
	}
	defer stateProducer.Close()

	// Subscribe to config messages (required for all modes, but joints config only needed for planar/dh)
	configPath := intentPath + ".config"
	configConsumer, err := bus.NewConsumer[*types.ManipulatorConfig](ctx, router, configPath)
	if err != nil {
		println("Failed to create config consumer:", err.Error())
		return
	}
	defer configConsumer.Close()

	// Subscribe to target messages
	targetPath := intentPath + ".target"
	targetConsumer, err := bus.NewConsumer[*types.ManipulatorTarget](ctx, router, targetPath)
	if err != nil {
		println("Failed to create target consumer:", err.Error())
		return
	}
	defer targetConsumer.Close()

	println("Manipulator initialized with intent:", intentPath)

	// Start config handler
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				config, err := configConsumer.Receive(ctx)
				if err != nil {
					println("Config receive error:", err.Error())
					continue
				}
				handleConfig(config)
			}
		}
	}()

	// Start target handler
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				target, err := targetConsumer.Receive(ctx)
				if err != nil {
					println("Target receive error:", err.Error())
					continue
				}
				handleTarget(target)
			}
		}
	}()

	// Main loop: update motion and publish state
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			state := updateMotion()
			if state != nil {
				if err := stateProducer.Send(ctx, state); err != nil {
					println("State send error:", err.Error())
				}
			}
			led.Set(!led.Get())
		}
	}
}

func setupRouter(ctx context.Context) (*dndm.Router, error) {
	// For TinyGo, we need to use the UART directly
	// This is a simplified setup - actual implementation may need adjustments for TinyGo compatibility
	embeddedPeer, err := dndm.PeerFromString("serial:///easyrobot.manipulator")
	if err != nil {
		return nil, err
	}

	// Create serial network node
	// Note: This may need adjustment for TinyGo compatibility
	serialNode, err := serial.New(embeddedPeer)
	if err != nil {
		return nil, err
	}

	// Create network factory
	factory, err := network.New(serialNode)
	if err != nil {
		return nil, err
	}

	// Create container endpoint that can accept endpoints dynamically
	containerEP := dndm.NewContainer("manipulator", 10)

	// Create router with container endpoint
	router, err := dndm.New(
		dndm.WithContext(ctx),
		dndm.WithQueueSize(10),
		dndm.WithEndpoint(containerEP),
	)
	if err != nil {
		return nil, err
	}

	// Serve (accept connections from host)
	// For serial, this opens the port and waits for host to connect
	go func() {
		err := factory.Serve(ctx, func(connectedPeer dndm.Peer, rwc io.ReadWriteCloser) error {
			// Create stream connection
			conn := stream.NewWithContext(ctx, embeddedPeer, connectedPeer, rwc, nil)

			// Create remote endpoint
			remoteEP := remote.New(embeddedPeer, conn, 10, time.Second*10, time.Second*3)
			err := remoteEP.Init(ctx, nil, // TinyGo may not have slog
				func(intent dndm.Intent, ep dndm.Endpoint) error { return nil },
				func(interest dndm.Interest, ep dndm.Endpoint) error { return nil },
			)
			if err != nil {
				return err
			}

			// Add remote endpoint to container
			if err := containerEP.Add(remoteEP); err != nil {
				println("Failed to add remote endpoint:", err.Error())
				return err
			}

			println("Connection established from:", connectedPeer.String())
			return nil
		})
		if err != nil {
			println("Serve error:", err.Error())
		}
	}()

	return router, nil
}

func handleConfig(config *types.ManipulatorConfig) {
	if config == nil {
		return
	}

	if len(config.Motors) != len(manipulatorConfig) {
		println("Motor count mismatch")
		return
	}

	// Convert proto motors to internal format
	motors := make([]servo.Motor, len(config.Motors))
	for i, m := range config.Motors {
		motors[i] = servo.NewMotorConfig(
			servo.WithPin(uint32(m.Pin)),
			servo.WithMicroseconds(m.MinUs, m.MaxUs, m.DefaultUs, m.MaxAngle),
		)
	}

	motionLock.Lock()
	defer motionLock.Unlock()

	if err := manipulator.Configure(motors); err != nil {
		println("Configure error:", err.Error())
		return
	}

	// Initialize motion controllers
	motion = NewMotion(len(motors))
	for i := range motors {
		motion[i].Input = motors[i].Default
		motion[i].Output = motors[i].Default
		motion[i].Target = motors[i].Default
	}

	// Handle kinematics configuration for planar/dh modes
	handleKinematicsConfig(config)

	println("Motors configured")
}

func handleTarget(target *types.ManipulatorTarget) {
	if target == nil {
		return
	}

	motionLock.Lock()
	defer motionLock.Unlock()

	// Check if we have kinematics (planar/dh mode)
	if kinematics != nil {
		// Kinematics mode: use end effector position
		if target.EndEffectorPosition == nil {
			println("End effector position required for kinematics mode")
			return
		}

		// Calculate joint angles using inverse kinematics
		jointAngles, err := kinematics.Inverse(
			target.EndEffectorPosition.X,
			target.EndEffectorPosition.Y,
			target.EndEffectorPosition.Z,
			target.EndEffectorOrientation,
		)
		if err != nil {
			println("IK error:", err.Error())
			return
		}

		// Convert from radians to degrees and set targets
		for i := range motion {
			if i < len(jointAngles) {
				// Convert radians to degrees
				motion[i].Target = jointAngles[i] * 180.0 / 3.14159265359
			}
		}
	} else {
		// Raw mode: use joint angles directly
		if len(target.JointAngles) != len(manipulatorConfig) {
			println("Target count mismatch")
			return
		}

		for i := range motion {
			if i < len(target.JointAngles) {
				motion[i].Target = target.JointAngles[i]
			}
		}
	}
}

func NewMotion(N int) []vaj.VAJ1D {
	m := make([]vaj.VAJ1D, N)
	for i := range m {
		m[i] = vaj.New1D(100, 100, 100)
	}
	return m
}

var now = time.Now()

func updateMotion() *types.ManipulatorState {
	state := make([]float32, len(motion))
	velocities := make([]float32, len(motion))

	motionLock.Lock()
	d := time.Since(now)
	for i := range motion {
		motion[i].Update(float32(d.Seconds()))
		state[i] = motion[i].Output
		velocities[i] = motion[i].Velocity
	}
	manipulator.Set(state[:len(motion)])
	motionLock.Unlock()

	now = time.Now()

	// Convert to proto state
	jointAngles := make([]float32, len(state))
	jointVelocities := make([]float32, len(velocities))
	copy(jointAngles, state)
	copy(jointVelocities, velocities)

	return &types.ManipulatorState{
		JointAngles:     jointAngles,
		JointVelocities: jointVelocities,
		Timestamp:       time.Now().UnixNano(),
	}
}
