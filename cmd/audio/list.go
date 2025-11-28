package main

import (
	"fmt"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/spf13/cobra"
)

func newListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List available audio devices",
		Long:  "Enumerate and display all available audio input (capture) and output (playback) devices.",
		RunE:  runList,
	}

	return cmd
}

func runList(cmd *cobra.Command, args []string) error {
	// Create unmarshaller to enumerate devices
	unmarshaller := audio.NewUnmarshaller()

	// Request device list by unmarshalling with "list" command
	var devices []types.AudioDeviceInfo
	reader := strings.NewReader("list")
	if err := unmarshaller.Unmarshal(reader, &devices); err != nil {
		return fmt.Errorf("failed to enumerate devices: %w", err)
	}

	// Separate devices by type
	var captureDevices []types.AudioDeviceInfo
	var playbackDevices []types.AudioDeviceInfo

	for _, device := range devices {
		switch device.DeviceType {
		case "capture":
			captureDevices = append(captureDevices, device)
		case "playback":
			playbackDevices = append(playbackDevices, device)
		}
	}

	// Display capture devices (inputs)
	fmt.Println("Input Devices (Capture):")
	if len(captureDevices) == 0 {
		fmt.Println("  (no input devices found)")
	} else {
		for i, device := range captureDevices {
			fmt.Printf("  [%d] %s", i+1, device.Name)
			if device.IsDefault {
				fmt.Print(" (default)")
			}
			fmt.Println()
			fmt.Printf("      ID: %s\n", device.ID)
			fmt.Printf("      Channels: %d-%d\n", device.MinChannels, device.MaxChannels)
			fmt.Printf("      Sample Rate: %d-%d Hz\n", device.MinSampleRate, device.MaxSampleRate)
			if len(device.Formats) > 0 {
				fmt.Printf("      Formats: %s\n", strings.Join(device.Formats, ", "))
			}
			fmt.Println()
		}
	}

	// Display playback devices (outputs)
	fmt.Println("Output Devices (Playback):")
	if len(playbackDevices) == 0 {
		fmt.Println("  (no output devices found)")
	} else {
		for i, device := range playbackDevices {
			fmt.Printf("  [%d] %s", i+1, device.Name)
			if device.IsDefault {
				fmt.Print(" (default)")
			}
			fmt.Println()
			fmt.Printf("      ID: %s\n", device.ID)
			fmt.Printf("      Channels: %d-%d\n", device.MinChannels, device.MaxChannels)
			fmt.Printf("      Sample Rate: %d-%d Hz\n", device.MinSampleRate, device.MaxSampleRate)
			if len(device.Formats) > 0 {
				fmt.Printf("      Formats: %s\n", strings.Join(device.Formats, ", "))
			}
			fmt.Println()
		}
	}

	return nil
}
