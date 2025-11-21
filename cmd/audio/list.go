package main

import (
	"fmt"

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
	// TODO: Implement device enumeration using audio unmarshaller
	// For now, return error indicating not yet implemented
	fmt.Println("Readers:")
	fmt.Println("  (device enumeration not yet implemented)")
	fmt.Println()
	fmt.Println("Writers:")
	fmt.Println("  (device enumeration not yet implemented)")

	return fmt.Errorf("device enumeration not yet implemented - Phase 4.1")
}
