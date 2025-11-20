# Display Command

Reads images, videos, or camera feeds and displays them using GoCV.

## Usage

```bash
go run ./cmd/display [flags]
```

## Source Options

Choose one of the following input sources:

### Images
```bash
--images /path/to/image.jpg
--images /path/to/image/directory
```

### Videos
```bash
--video /path/to/video.mp4
```

### Cameras (Direct)
```bash
--camera 0 --width 640 --height 480
--camera 0 --camera 1 --width 1024 --height 768  # Multiple cameras
```

### Camera Enumeration (Interactive)
```bash
--enumerate-cameras
```

When using `--enumerate-cameras`, the command will:
1. List all available camera devices
2. Allow you to select which cameras to use
3. Prompt for resolution, pixel format, and frame rate
4. Start capture with your selected configuration

## Examples

### Enumerate and configure cameras interactively
```bash
./display --enumerate-cameras
```
Output:
```
=== Camera Enumeration ===
Found 2 camera(s):
  [0] Camera 0 (/dev/video0)
      Driver: v4l2
      Formats: 2 available
      Example format: 640x480
      Controls: 8 available

  [1] Camera 1 (/dev/video1)
      Driver: v4l2
      Formats: 2 available
      Example format: 640x480
      Controls: 8 available

Enter camera ID(s) to use (comma-separated, or 'all' for all cameras): 0,1
Enter resolution (width x height, default 640x480): 1024x768
Enter pixel format (default: MJPEG, options: MJPEG, YUYV, etc.): YUYV
Enter frame rate (default: 30): 30

Selected configuration:
  Cameras: [0 1]
  Resolution: 1024x768
  Pixel Format: YUYV
  Frame Rate: 30 fps
```

### Display images from directory
```bash
./display --images /path/to/images/
```

### Display video file
```bash
./display --video input.mp4
```

### Display from specific cameras
```bash
./display --camera 0 --width 1920 --height 1080
```

## Destination Options

The display command supports various output destinations. See the main help for available options:

```bash
./display --help
```