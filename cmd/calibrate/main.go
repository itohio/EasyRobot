package main

import (
	"os"

	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "calibrate",
		Short: "Camera calibration tool",
		Long:  "Calibrate cameras using checkerboard patterns. Supports monocular and stereo camera calibration.",
	}

	// Add subcommands
	rootCmd.AddCommand(newCamMonoCmd())
	rootCmd.AddCommand(newCamStereoCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

