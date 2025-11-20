// go:build linux
package v4l_test

import (
	"fmt"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/marshaller/v4l"
)

func ExampleUnmarshaller_Unmarshal_deviceList() {
	// Create V4L unmarshaller
	unmarshaller := v4l.NewUnmarshaller()
	if err != nil {
		fmt.Printf("Failed to create unmarshaller: %v\n", err)
		return
	}

	// Enumerate devices
	var devices []types.CameraInfo
	err = unmar.Unmarshal(strings.NewReader("list"), &devices)
	if err != nil {
		fmt.Printf("Failed to list devices: %v\n", err)
		return
	}

	for _, dev := range devices {
		fmt.Printf("Camera: %s (%s)\n", dev.Name, dev.Path)
		fmt.Printf("  Formats: %d available\n", len(dev.SupportedFormats))
		fmt.Printf("  Controls: %d available\n", len(dev.Controls))
	}
}

func ExampleUnmarshaller_Unmarshal_singleStream() {
	// Configure camera via unmarshaller options
	unmarshaller := v4l.NewUnmarshaller(
		// Camera 0: 640x480 MJPEG at 30fps
		v4l.WithVideoDeviceEx(0, 640, 480, 30, "MJPEG"),

		// Configure camera controls
		v4l.WithCameraControls(map[string]int32{
			types.CameraControlBrightness: 128,
			types.CameraControlContrast:   32,
		}),
	)

	// Create frame stream
	var stream types.FrameStream
	err := unmarshaller.Unmarshal(nil, &stream)
	if err != nil {
		fmt.Printf("Failed to create stream: %v\n", err)
		return
	}
	defer stream.Close()

	// Capture a few frames
	framesCaptured := 0
	for frame := range stream.C {
		fmt.Printf("Captured frame %d: %v\n", frame.Index, frame.Tensors[0].Shape())
		frame.Tensors[0].Release()
		framesCaptured++

		if framesCaptured >= 3 {
			break
		}
	}

	fmt.Println("Capture completed successfully")
}

func ExampleUnmarshaller_Unmarshal_multiStream() {
	// Configure dual camera capture
	unmarshaller := v4l.NewUnmarshaller(
		// Camera 0: 640x480 YUV at 30fps
		v4l.WithVideoDeviceEx(0, 640, 480, 30, "YUYV"),

		// Camera 1: 640x480 YUV at 30fps
		v4l.WithVideoDeviceEx(1, 640, 480, 30, "YUYV"),

		// Configure controls for both cameras
		v4l.WithCameraControls(map[string]int32{
			types.CameraControlBrightness: 128,
			types.CameraControlContrast:   32,
		}),
	)

	// Create synchronized frame stream
	var stream types.FrameStream
	err := unmarshaller.Unmarshal(nil, &stream)
	if err != nil {
		fmt.Printf("Failed to create stream: %v\n", err)
		return
	}
	defer stream.Close()

	// Capture synchronized frames
	framesCaptured := 0
	for frame := range stream.C {
		fmt.Printf("Captured synchronized frame %d with %d tensors\n",
			frame.Index, len(frame.Tensors))

		// Process each camera's tensor
		for i, tensor := range frame.Tensors {
			fmt.Printf("  Camera %d: %v\n", i, tensor.Shape())

			// Access individual camera controls
			devicePath := fmt.Sprintf("/dev/video%d", i)
			if controller := unmarshaller.CameraController(devicePath); controller != nil {
				exposure, _ := controller.GetControl(types.CameraControlExposure)
				fmt.Printf("    Exposure: %d\n", exposure)
			}

			tensor.Release()
		}
		framesCaptured++

		if framesCaptured >= 2 {
			break
		}
	}

	fmt.Println("Synchronized capture completed successfully")
}

func ExampleDevice_Open() {
	// Open device directly
	device, err := v4l.NewDevice("/dev/video0")
	if err != nil {
		fmt.Printf("Failed to open device: %v\n", err)
		return
	}
	defer device.Close()

	// Get device info
	info := device.Info()
	fmt.Printf("Opened camera: %s (%s)\n", info.Name, info.Path)

	// Open stream with shared options
	stream, err := device.Open(
		types.WithCameraResolution(1280, 720),
		types.WithCameraPixelFormat("MJPEG"),
		types.WithCameraBufferCount(8),
	)
	if err != nil {
		fmt.Printf("Failed to open stream: %v\n", err)
		return
	}
	defer stream.Close()

	// Check available controls via controller
	controller := stream.Controller()
	controls := controller.Controls()
	fmt.Printf("Available controls: %d\n", len(controls))

	for _, ctrl := range controls {
		fmt.Printf("  %s: %d-%d (default: %d)\n", ctrl.Name, ctrl.Min, ctrl.Max, ctrl.Default)
	}
}

func ExampleUnmarshaller_CameraController() {
	// Configure camera
	unmarshaller := v4l.NewUnmarshaller(
		v4l.WithVideoDeviceEx(0, 640, 480, 30, "MJPEG"),
	)

	// Create stream to initialize controllers
	var stream types.FrameStream
	err := unmarshaller.Unmarshal(nil, &stream)
	if err != nil {
		fmt.Printf("Failed to create stream: %v\n", err)
		return
	}
	defer stream.Close()

	// Get camera controller from unmarshaller
	controller := unmarshaller.CameraController("/dev/video0")
	if controller == nil {
		fmt.Println("Controller not available")
		return
	}

	// Set brightness control
	err = controller.SetControl(types.CameraControlBrightness, 128)
	if err != nil {
		fmt.Printf("Failed to set brightness: %v\n", err)
		return
	}

	// Get brightness control
	brightness, err := controller.GetControl(types.CameraControlBrightness)
	if err != nil {
		fmt.Printf("Failed to get brightness: %v\n", err)
		return
	}

	fmt.Printf("Brightness set to: %d\n", brightness)

	// Set multiple controls
	err = controller.SetControls(map[string]int32{
		types.CameraControlContrast:   32,
		types.CameraControlSaturation: 64,
	})
	if err != nil {
		fmt.Printf("Failed to set multiple controls: %v\n", err)
		return
	}

	fmt.Println("Multiple controls set successfully")
}

func ExampleGetAllDevices() {
	devices, err := v4l.GetAllDevices()
	if err != nil {
		fmt.Printf("Failed to get devices: %v\n", err)
		return
	}

	fmt.Printf("Found %d V4L devices:\n", len(devices))
	for i, device := range devices {
		info := device.Info()
		fmt.Printf("  %d: %s (%s)\n", i, info.Name, info.Path)
		fmt.Printf("     Formats: %d available\n", len(info.SupportedFormats))
		fmt.Printf("     Controls: %d available\n", len(info.Controls))
	}
}
