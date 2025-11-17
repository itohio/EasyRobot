//go:build linux || windows

package main

import "fmt"

// Config holds configuration for the LiDAR driver.
type Config struct {
	SerialPort   string
	LidarType    string
	TargetPts    int
	Display      bool
	WindowWidth  int
	WindowHeight int
	ImageWidth   int
	ImageHeight  int
	ScaleFactor  float64
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	if c.SerialPort == "" {
		return fmt.Errorf("serial port required")
	}
	if c.Display {
		if c.ImageWidth <= 0 || c.ImageHeight <= 0 {
			return fmt.Errorf("image dimensions must be positive")
		}
		if c.ScaleFactor <= 0 {
			return fmt.Errorf("scale factor must be positive")
		}
		if c.WindowWidth < 0 || c.WindowHeight < 0 {
			return fmt.Errorf("window dimensions must be non-negative")
		}
	}
	return nil
}

// WindowSize returns the effective window size (uses image size if not specified).
func (c *Config) WindowSize() (width, height int) {
	width = c.WindowWidth
	height = c.WindowHeight
	if width == 0 {
		width = c.ImageWidth
	}
	if height == 0 {
		height = c.ImageHeight
	}
	return width, height
}

