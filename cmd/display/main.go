package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
)

func main() {
	// Register flags
	source.RegisterAllFlags()
	destination.RegisterAllFlags()
	// Register DNDM flags if needed (currently optional)
	source.RegisterInterestFlags()
	destination.RegisterIntentFlags()
	help := flag.Bool("help", false, "Show help message")

	// Parse flags
	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	// Create source from flags
	src, err := source.NewFromFlags()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Create destinations from flags
	dests := destination.NewAllDestinations()

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
	}()

	// Start source
	if err := src.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting source: %v\n", err)
		os.Exit(1)
	}
	defer src.Close()

	// Start destinations
	for _, dest := range dests {
		if err := dest.Start(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Error starting destination: %v\n", err)
			os.Exit(1)
		}
		defer dest.Close()
	}

	// Process frames
	for {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Read frame from source
		frame, err := src.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				// Source exhausted, exit gracefully
				return
			}
			fmt.Fprintf(os.Stderr, "Error reading frame: %v\n", err)
			// Continue processing if error is recoverable
			continue
		}

		// Send frame to all destinations
		for _, dest := range dests {
			if err := dest.AddFrame(frame); err != nil {
				fmt.Fprintf(os.Stderr, "Error adding frame to destination: %v\n", err)
				// Continue to other destinations
			}
		}

		// Check if display was closed (ESC key pressed)
		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}
