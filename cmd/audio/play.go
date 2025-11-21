package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

func newPlayCmd() *cobra.Command {
	var (
		input    string
		deviceID string
	)

	cmd := &cobra.Command{
		Use:   "play",
		Short: "Play WAV file to audio device",
		Long:  "Read a WAV file and play it to a playback device.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runPlay(input, deviceID)
		},
	}

	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file path (required)")
	cmd.Flags().StringVar(&deviceID, "device-id", "", "Playback device ID (optional, uses default if not specified)")

	cmd.MarkFlagRequired("input")

	return cmd
}

func runPlay(input, deviceID string) error {
	// TODO: Implement playback using audio unmarshaller and marshaller
	// For now, return error indicating not yet implemented

	fmt.Println("Output devices:")
	fmt.Println("  (device enumeration not yet implemented)")
	fmt.Println()

	if input == "" {
		return fmt.Errorf("input file path is required")
	}

	// Set up signal handling for Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("Ctrl+C to stop")
	<-sigChan

	fmt.Println("Playback stopped")

	return fmt.Errorf("playback not yet implemented - Phase 5.1")
}
