package ports

import (
	"context"
	"fmt"

	"github.com/itohio/EasyRobot/x/devices"
)

// Run lists all available serial ports on the system.
// Reuses devices.ListSerialPorts() for port enumeration.
func Run(ctx context.Context) error {
	ports, err := devices.ListSerialPorts()
	if err != nil {
		return fmt.Errorf("failed to list serial ports: %w", err)
	}

	if len(ports) == 0 {
		fmt.Println("No serial ports found.")
		return nil
	}

	fmt.Printf("Available serial ports (%d):\n\n", len(ports))
	for _, port := range ports {
		fmt.Printf("  %s\n", port)
	}
	return nil
}
