package main

import (
	"os"

	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "audio",
		Short: "Audio marshaller/unmarshaller command-line tool",
		Long:  "Command-line tool for audio operations: listing devices, recording, playing, and generating audio signals.",
	}

	// Add subcommands
	rootCmd.AddCommand(newListCmd())
	rootCmd.AddCommand(newRecordCmd())
	rootCmd.AddCommand(newPlayCmd())
	rootCmd.AddCommand(newGenSineCmd())
	rootCmd.AddCommand(newGenNoiseCmd())
	rootCmd.AddCommand(newGenChirpCmd())
	rootCmd.AddCommand(newGenPSKCmd())
	rootCmd.AddCommand(newGenFSKCmd())
	rootCmd.AddCommand(newGenQAMCmd())
	rootCmd.AddCommand(newGenDSSSCmd())
	rootCmd.AddCommand(newGenMASKCmd())
	rootCmd.AddCommand(newGenTCCmd())
	rootCmd.AddCommand(newRecvPSKCmd())
	rootCmd.AddCommand(newRecvFSKCmd())
	rootCmd.AddCommand(newRecvQAMCmd())
	rootCmd.AddCommand(newRecvDSSSCmd())
	rootCmd.AddCommand(newRecvMASKCmd())
	rootCmd.AddCommand(newRecvTCCmd())
	rootCmd.AddCommand(newRecvChirpCmd())
	rootCmd.AddCommand(newMixCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
