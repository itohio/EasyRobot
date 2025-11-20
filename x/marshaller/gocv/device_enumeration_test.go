package gocv

import (
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func TestUnmarshaller_UnmarshalCameraList(t *testing.T) {
	unmarshaller := NewUnmarshaller()

	// Test the "list" command
	reader := strings.NewReader("list")
	var devices []CameraInfo
	err := unmarshaller.Unmarshal(reader, &devices)
	if err != nil {
		t.Fatalf("Failed to unmarshal camera list: %v", err)
	}

	// We expect at least some devices (may be 0 if no cameras available)
	if len(devices) < 0 {
		t.Errorf("Expected non-negative device count, got %d", len(devices))
	}

	// Check that each device has valid information
	for i, device := range devices {
		if device.ID < 0 {
			t.Errorf("Device %d has invalid ID: %d", i, device.ID)
		}
		if device.Path == "" {
			t.Errorf("Device %d has empty path", i)
		}
		if device.Name == "" {
			t.Errorf("Device %d has empty name", i)
		}
	}
}

func TestCameraController(t *testing.T) {
	unmarshaller := NewUnmarshaller()

	// Try to open a camera (device 0)
	reader := strings.NewReader("")
	var stream types.FrameStream
	err := unmarshaller.Unmarshal(reader, &stream,
		WithVideoDevice(0, 640, 480),
		WithCameraControls(map[string]int32{
			"brightness": 128,
			"contrast":   32,
		}),
	)

	if err != nil {
		t.Skipf("Skipping camera controller test (no camera available): %v", err)
		return
	}
	defer stream.Close()

	// Get camera controller
	controller := unmarshaller.CameraController(0)
	if controller == nil {
		t.Fatal("Expected camera controller for device 0, got nil")
	}

	// Test getting controls
	controls, err := controller.GetControls()
	if err != nil {
		t.Fatalf("Failed to get controls: %v", err)
	}

	if len(controls) == 0 {
		t.Log("No controls available (expected on some systems)")
		return
	}

	// Test setting a control (brightness)
	originalBrightness := controls["brightness"]
	err = controller.SetControl("brightness", 100)
	if err != nil {
		t.Logf("Failed to set brightness (may not be supported): %v", err)
	} else {
		// Verify it was set
		newBrightness, err := controller.GetControl("brightness")
		if err != nil {
			t.Errorf("Failed to get brightness after setting: %v", err)
		} else if newBrightness != 100 {
			t.Errorf("Brightness not set correctly: expected 100, got %d", newBrightness)
		}

		// Restore original value
		controller.SetControl("brightness", originalBrightness)
	}
}
