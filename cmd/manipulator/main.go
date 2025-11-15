package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/itohio/dndm"
	"github.com/itohio/dndm/endpoint/remote"
	"github.com/itohio/dndm/network"
	dndm_serial "github.com/itohio/dndm/network/serial"
	"github.com/itohio/dndm/network/stream"
	"github.com/itohio/dndm/x/bus"

	types "github.com/itohio/EasyRobot/types/control"
	types_kinematics "github.com/itohio/EasyRobot/types/control/kinematics"
	types_math "github.com/itohio/EasyRobot/types/math"
)

func main() {
	help := flag.Bool("help", false, "Show help message")
	listPorts := flag.Bool("list", false, "List all serial ports")
	port := flag.String("port", "/dev/ttyACM0", "Serial port path")
	baud := flag.Int("baud", 115200, "Serial port baud rate")
	mode := flag.String("mode", "raw", "Build mode: raw, planar, or dh")

	flag.Parse()

	if *help {
		fmt.Println("Manipulator Client")
		fmt.Println("Connects to manipulator driver on XIAO board via DNDM")
		fmt.Println()
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Commands:")
		fmt.Println("  config - send manipulator configuration")
		fmt.Println("  target <joint1> [joint2] [joint3] - set joint angle targets (raw mode)")
		fmt.Println("  pose <x> <y> <z> [w] [x] [y] [z] - set end effector pose (planar/dh mode)")
		fmt.Println("  state - show last received state")
		fmt.Println("  quit - exit")
		return
	}

	if *listPorts {
		ports, err := listSerialPorts()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing ports: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("Available serial ports:")
		for i, p := range ports {
			fmt.Printf("%d\t%s\n", i, p)
		}
		return
	}

	// Validate mode
	intentPath := getIntentPath(*mode)
	if intentPath == "" {
		fmt.Fprintf(os.Stderr, "Invalid mode: %s (must be raw, planar, or dh)\n", *mode)
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// Setup DNDM router with serial endpoint
	router, err := setupRouter(ctx, *port, *baud)
	if err != nil {
		slog.Error("Failed to setup router", "err", err)
		os.Exit(1)
	}
	defer router.Close()

	slog.Info("Manipulator client starting", "mode", *mode, "intent", intentPath, "port", *port, "baud", *baud)

	// Subscribe to state (Interest - receives from driver's Intent)
	stateConsumer, err := bus.NewConsumer[*types.ManipulatorState](ctx, router, intentPath)
	if err != nil {
		slog.Error("Failed to create state consumer", "err", err)
		os.Exit(1)
	}
	defer stateConsumer.Close()

	// Create config producer (Intent - sends to driver's Interest)
	configPath := intentPath + ".config"
	configProducer, err := bus.NewProducer[*types.ManipulatorConfig](ctx, router, configPath)
	if err != nil {
		slog.Error("Failed to create config producer", "err", err)
		os.Exit(1)
	}
	defer configProducer.Close()

	// Create target producer (Intent - sends to driver's Interest)
	targetPath := intentPath + ".target"
	targetProducer, err := bus.NewProducer[*types.ManipulatorTarget](ctx, router, targetPath)
	if err != nil {
		slog.Error("Failed to create target producer", "err", err)
		os.Exit(1)
	}
	defer targetProducer.Close()

	slog.Info("Client components created", "mode", *mode)

	// State tracking
	var lastState *types.ManipulatorState
	stateLock := make(chan *types.ManipulatorState, 1)

	// Start state receiver goroutine
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				state, err := stateConsumer.Receive(ctx)
				if err != nil {
					slog.Error("Failed to receive state", "err", err)
					continue
				}

				// Update last state
				select {
				case stateLock <- state:
					lastState = state
				default:
					lastState = state
				}

				slog.Info("Received state",
					"joint_angles", state.JointAngles,
					"timestamp", state.Timestamp)
			}
		}
	}()

	// Wait a bit for connection to establish
	time.Sleep(time.Second)

	// Start interactive command handler
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("Manipulator Client - Interactive Mode")
	fmt.Printf("Mode: %s\n", *mode)
	fmt.Printf("Intent path: %s\n", intentPath)
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  config - send manipulator configuration")
	if *mode == "raw" {
		fmt.Println("  target <joint1> [joint2] [joint3] - set joint angle targets (degrees)")
	} else {
		fmt.Println("  pose <x> <y> <z> [w] [x] [y] [z] - set end effector pose (position in m, orientation as quaternion)")
	}
	fmt.Println("  state - show last received state")
	fmt.Println("  quit - exit")
	fmt.Println()

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		switch parts[0] {
		case "config":
			if err := sendConfig(ctx, configProducer, *mode); err != nil {
				slog.Error("Failed to send config", "err", err)
			} else {
				slog.Info("Config sent")
			}

		case "target":
			if *mode != "raw" {
				fmt.Printf("Target command only works in raw mode. Use 'pose' for %s mode.\n", *mode)
				continue
			}
			if err := sendTarget(ctx, targetProducer, parts[1:]); err != nil {
				slog.Error("Failed to send target", "err", err)
			} else {
				slog.Info("Target sent")
			}

		case "pose":
			if *mode == "raw" {
				fmt.Println("Pose command only works in planar/dh mode. Use 'target' for raw mode.")
				continue
			}
			if err := sendPose(ctx, targetProducer, parts[1:]); err != nil {
				slog.Error("Failed to send pose", "err", err)
			} else {
				slog.Info("Pose sent")
			}

		case "state":
			if lastState != nil {
				fmt.Printf("Joint angles: %v (degrees)\n", lastState.JointAngles)
				if len(lastState.JointVelocities) > 0 {
					fmt.Printf("Joint velocities: %v (deg/s)\n", lastState.JointVelocities)
				}
				fmt.Printf("Timestamp: %d ns\n", lastState.Timestamp)
			} else {
				fmt.Println("No state received yet")
			}

		case "quit":
			cancel()
			return

		default:
			fmt.Printf("Unknown command: %s\n", parts[0])
		}
	}

	<-ctx.Done()
	slog.Info("Client stopped")
}

func getIntentPath(mode string) string {
	switch mode {
	case "raw":
		return "easyrobot.manipulator.raw"
	case "planar":
		return "easyrobot.manipulator.planar"
	case "dh":
		return "easyrobot.manipulator.dh"
	default:
		return ""
	}
}

func setupRouter(ctx context.Context, port string, baud int) (*dndm.Router, error) {
	// Host peer (connecting to serial)
	hostPeer, err := dndm.PeerFromString(fmt.Sprintf("serial://%s/easyrobot.host?baud=%d", port, baud))
	if err != nil {
		return nil, err
	}

	// Embedded device peer
	embeddedPeer, err := dndm.PeerFromString(fmt.Sprintf("serial://%s/easyrobot.manipulator?baud=%d", port, baud))
	if err != nil {
		return nil, err
	}

	// Create serial network node
	serialNode, err := dndm_serial.New(hostPeer)
	if err != nil {
		return nil, err
	}

	// Create network factory
	factory, err := network.New(serialNode)
	if err != nil {
		return nil, err
	}

	// Dial the serial connection
	rwc, err := factory.Dial(ctx, embeddedPeer)
	if err != nil {
		return nil, err
	}

	// Create stream connection
	conn := stream.NewWithContext(ctx, hostPeer, embeddedPeer, rwc, nil)

	// Create remote endpoint
	remoteEP := remote.New(hostPeer, conn, 10, time.Second*10, time.Second*3)
	err = remoteEP.Init(ctx, slog.Default(),
		func(intent dndm.Intent, ep dndm.Endpoint) error { return nil },
		func(interest dndm.Interest, ep dndm.Endpoint) error { return nil },
	)
	if err != nil {
		return nil, err
	}

	// Create router with remote endpoint
	router, err := dndm.New(
		dndm.WithContext(ctx),
		dndm.WithQueueSize(10),
		dndm.WithEndpoint(remoteEP),
	)
	if err != nil {
		return nil, err
	}

	return router, nil
}

func sendConfig(ctx context.Context, producer *bus.Producer[*types.ManipulatorConfig], mode string) error {
	config := &types.ManipulatorConfig{
		Motors: []*types.ServoMotorConfig{
			{
				Pin:          8,
				MinUs:        500,
				MaxUs:        2500,
				DefaultUs:    1500,
				MaxAngle:     180,
				DefaultAngle: 90,
			},
			{
				Pin:          9,
				MinUs:        500,
				MaxUs:        2500,
				DefaultUs:    1500,
				MaxAngle:     180,
				DefaultAngle: 90,
			},
			{
				Pin:          10,
				MinUs:        500,
				MaxUs:        2500,
				DefaultUs:    1500,
				MaxAngle:     180,
				DefaultAngle: 90,
			},
		},
		Motion: []*types.Constraints{
			{MaxSpeed: 50, MaxAcceleration: 10, MaxJerk: 1},
			{MaxSpeed: 50, MaxAcceleration: 10, MaxJerk: 1},
			{MaxSpeed: 50, MaxAcceleration: 10, MaxJerk: 1},
		},
	}

	// Add kinematics config for planar/dh modes
	if mode == "planar" {
		config.Joints = &types_kinematics.JointsConfig{
			PlanarJoints: []*types_kinematics.PlanarJoint{
				{MinAngle: -90, MaxAngle: 90, Length: 39},
				{MinAngle: -90, MaxAngle: 90, Length: 45},
				{MinAngle: -10, MaxAngle: 90, Length: 100},
			},
		}
	} else if mode == "dh" {
		// TODO: Add DH parameters when available
		config.Joints = &types_kinematics.JointsConfig{
			// DhParams would go here
		}
	}

	return producer.Send(ctx, config)
}

func sendTarget(ctx context.Context, producer *bus.Producer[*types.ManipulatorTarget], args []string) error {
	if len(args) < 1 || len(args) > 3 {
		return fmt.Errorf("target requires 1-3 joint angles (degrees)")
	}

	jointAngles := make([]float32, 0, 3)
	for _, arg := range args {
		angle, err := strconv.ParseFloat(arg, 32)
		if err != nil {
			return fmt.Errorf("invalid angle: %s", arg)
		}
		jointAngles = append(jointAngles, float32(angle))
	}

	target := &types.ManipulatorTarget{
		JointAngles: jointAngles,
	}

	return producer.Send(ctx, target)
}

func sendPose(ctx context.Context, producer *bus.Producer[*types.ManipulatorTarget], args []string) error {
	if len(args) < 3 {
		return fmt.Errorf("pose requires at least x, y, z (meters)")
	}

	x, err := strconv.ParseFloat(args[0], 32)
	if err != nil {
		return fmt.Errorf("invalid x: %s", args[0])
	}

	y, err := strconv.ParseFloat(args[1], 32)
	if err != nil {
		return fmt.Errorf("invalid y: %s", args[1])
	}

	z, err := strconv.ParseFloat(args[2], 32)
	if err != nil {
		return fmt.Errorf("invalid z: %s", args[2])
	}

	target := &types.ManipulatorTarget{
		EndEffectorPosition: &types_math.Vector3D{
			X: float32(x),
			Y: float32(y),
			Z: float32(z),
		},
	}

	// Optional orientation (quaternion)
	if len(args) >= 7 {
		w, err := strconv.ParseFloat(args[3], 32)
		if err != nil {
			return fmt.Errorf("invalid quaternion w: %s", args[3])
		}
		qx, err := strconv.ParseFloat(args[4], 32)
		if err != nil {
			return fmt.Errorf("invalid quaternion x: %s", args[4])
		}
		qy, err := strconv.ParseFloat(args[5], 32)
		if err != nil {
			return fmt.Errorf("invalid quaternion y: %s", args[5])
		}
		qz, err := strconv.ParseFloat(args[6], 32)
		if err != nil {
			return fmt.Errorf("invalid quaternion z: %s", args[6])
		}

		target.EndEffectorOrientation = &types_math.Quaternion{
			W: float32(w),
			X: float32(qx),
			Y: float32(qy),
			Z: float32(qz),
		}
	}

	return producer.Send(ctx, target)
}

// listSerialPorts lists available serial ports using standard library file operations.
// On Linux, it searches /dev for ttyACM*, ttyUSB*, ttyS*, and similar devices.
func listSerialPorts() ([]string, error) {
	var ports []string

	// Common serial port patterns on Linux
	patterns := []string{
		"/dev/ttyACM*", // USB serial (e.g., Arduino)
		"/dev/ttyUSB*", // USB serial converters
		"/dev/ttyS*",   // Serial ports
	}

	seen := make(map[string]bool)

	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			continue
		}
		for _, match := range matches {
			if !seen[match] {
				// Check if it's actually a character device
				info, err := os.Stat(match)
				if err == nil && (info.Mode()&os.ModeCharDevice) != 0 {
					ports = append(ports, match)
					seen[match] = true
				}
			}
		}
	}

	sort.Strings(ports)
	return ports, nil
}
