//go:build rp2040

// NOTE: This package is designed for building only on TinyGo for RP2040 microcontroller.
// Build tags determine the LiDAR type:
//   - xwpftb tag: XWPFTB LiDAR
//   - ld06 tag: LD06 LiDAR
// Usage:
//   - XWPFTB: tinygo flash -target=rp2040 -tags xwpftb .
//   - LD06: tinygo flash -target=rp2040 -tags ld06 .

package main

import (
	"context"
	"io"
	"machine"
	"time"

	"github.com/itohio/dndm"
	"github.com/itohio/dndm/endpoint/remote"
	"github.com/itohio/dndm/network"
	"github.com/itohio/dndm/network/serial"
	"github.com/itohio/dndm/network/stream"
	"github.com/itohio/dndm/x/bus"

	pbdevices "github.com/itohio/EasyRobot/types/devices"
	devio "github.com/itohio/EasyRobot/x/devices"
	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

// Hardware configuration - adjust these for your board
const (
	uartTX = machine.GPIO0 // Adjust for your board
	uartRX = machine.GPIO1 // Adjust for your board
	pwmPin = machine.GPIO2 // Adjust for your board (optional, set to 0 to disable)
)

// LiDAR configuration
const (
	targetPoints = 0 // 0 = auto-calibrate
	maxPoints    = 2048
)

var (
	uart = machine.UART0
)

func main() {
	ctx := context.Background()

	// Configure UART
	uart.Configure(machine.UARTConfig{
		BaudRate: getBaudRate(),
		TX:       uartTX,
		RX:       uartRX,
	})
	// For TinyGo, machine.UART implements io.Reader/Writer directly
	// Create a simple wrapper that implements the Serial interface
	ser := &tinyGoSerialWrapper{uart: uart}

	// Setup PWM if pin is configured (optional)
	var motor devio.PWM
	if pwmPin != 0 {
		// For RP2040, PWM setup would go here
		// This is a placeholder - actual implementation depends on RP2040 PWM support
		println("PWM support for RP2040 not yet implemented")
	}

	// Setup DNDM router
	router, err := setupRouter(ctx)
	if err != nil {
		println("Failed to setup router:", err.Error())
		return
	}
	defer router.Close()

	// Create LiDAR reading producer
	intentPath := "LIDARReading@lidar.scan"
	producer, err := bus.NewProducer[*pbdevices.LIDARReading](ctx, router, intentPath)
	if err != nil {
		println("Failed to create producer:", err.Error())
		return
	}
	defer producer.Close()

	println("LiDAR driver initialized with intent:", intentPath)

	// Create LiDAR device based on build tag
	lidar := createLIDAR(ctx, ser, motor)
	if lidar == nil {
		println("Failed to create LiDAR device")
		return
	}
	defer lidar.Close()

	if err := lidar.Configure(true); err != nil {
		println("Failed to configure LiDAR:", err.Error())
		return
	}

	// Register callback to publish readings
	lidar.OnRead(func(m matTypes.Matrix) {
		reading := matrixToLIDARReading(m)
		if err := producer.Send(ctx, reading); err != nil {
			println("Failed to send reading:", err.Error())
		}
	})

	// Keep running
	for {
		select {
		case <-ctx.Done():
			return
		default:
			time.Sleep(100 * time.Millisecond)
		}
	}
}

type lidarDevice interface {
	Configure(init bool) error
	OnRead(fn func(matTypes.Matrix))
	Close()
}

// tinyGoSerialWrapper wraps machine.UART to implement devio.Serial interface
type tinyGoSerialWrapper struct {
	uart machine.UART
}

func (s *tinyGoSerialWrapper) Read(p []byte) (n int, err error) {
	return s.uart.Read(p)
}

func (s *tinyGoSerialWrapper) Write(p []byte) (n int, err error) {
	return s.uart.Write(p)
}

func (s *tinyGoSerialWrapper) Buffered() int {
	return s.uart.Buffered()
}

// createLIDAR creates the appropriate LiDAR device based on build tags
func createLIDAR(ctx context.Context, ser devio.Serial, motor devio.PWM) lidarDevice {
	return createLIDARImpl(ctx, ser, motor)
}

// getBaudRate returns the baud rate based on LiDAR type
func getBaudRate() uint32 {
	return getBaudRateImpl()
}

func setupRouter(ctx context.Context) (*dndm.Router, error) {
	embeddedPeer, err := dndm.PeerFromString("serial:///easyrobot.lidar")
	if err != nil {
		return nil, err
	}

	serialNode, err := serial.New(embeddedPeer)
	if err != nil {
		return nil, err
	}

	factory, err := network.New(serialNode)
	if err != nil {
		return nil, err
	}

	// Create container endpoint that can accept endpoints dynamically
	containerEP := dndm.NewContainer("lidar", 10)

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
			err := remoteEP.Init(ctx, nil,
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

func matrixToLIDARReading(m matTypes.Matrix) *pbdevices.LIDARReading {
	if m == nil || m.Cols() == 0 {
		return &pbdevices.LIDARReading{
			Valid: false,
		}
	}

	view := m.View().(mat.Matrix)
	distances := make([]float32, m.Cols())
	angles := make([]float32, m.Cols())

	for i := 0; i < m.Cols(); i++ {
		distances[i] = view[0][i]
		angles[i] = view[1][i]
	}

	return &pbdevices.LIDARReading{
		DistancesMm: distances,
		AnglesDeg:   angles,
		Timestamp:   time.Now().UnixNano(),
		Valid:       true,
	}
}
