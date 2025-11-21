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
# Simple: just camera ID (uses default width/height from --width/--height flags)
--camera 0

# With resolution
--camera 0:640x480

# With resolution and frame rate
--camera 0:640x480@30

# With resolution, frame rate, and pixel format
--camera 0:640x480@30/mjpeg    # MJPEG compressed format (common, good for high res/fps)
--camera 0:640x480@30/yuyv     # YUYV raw format (uncompressed)
--camera 0:640x480@30/yuv2      # UYVY raw format (uncompressed)
--camera 0:640x480@30/bgr       # BGR24 raw format (uncompressed)

# Multiple cameras with different configurations
--camera 0:640x480@30/mjpeg --camera 1:1280x720@60/bgr

# Legacy format (still supported): use --width and --height flags
--camera 0 --width 640 --height 480
```

Camera configuration format: `ID[:widthxheight[@fps[/format]]]`
- `ID` - Camera device ID (required)
- `widthxheight` - Resolution (optional, e.g., `640x480`)
- `fps` - Frame rate (optional, e.g., `@30`)
- `format` - Pixel format (optional, e.g., `/mjpeg`, `/yuyv`, `/bgr`)

All components except ID are optional. If not specified, defaults from `--width`, `--height`, and other flags are used.

**Supported Pixel Formats:**
- `mjpeg`, `mjpg`, `jpeg` - MJPEG compressed (good for high resolution/fps, requires decoding)
- `yuyv`, `yuy2` - YUYV raw format (uncompressed)
- `yuv2`, `uyvy` - UYVY raw format (uncompressed)
- `rgb24`, `rgb3` - RGB24 raw format (uncompressed)
- `bgr24`, `bgr3` - BGR24 raw format (uncompressed)
- `nv12` - NV12 format
- `nv21` - NV21 format
- `i420`, `yv12` - I420/YV12 format
- `h264` - H.264 compressed
- `h265`, `hevc` - H.265/HEVC compressed
- Or any 4-character FOURCC code (e.g., `GREY`, `RGGB`)

**Note:** Not all cameras support all formats. Use `--list-cameras` to see which formats your camera supports. If a format is not supported, the camera will use its default format.

### List Cameras
```bash
--list-cameras
```

When using `--list-cameras`, the command will:
1. List all available camera devices with their complete parameters
2. Show all supported formats for each camera
3. Show all available controls with their ranges and current values
4. Exit immediately after listing (does not start streaming)

Use this to discover available cameras and their capabilities before using `--camera` to start streaming.

## Examples

### List all cameras and their parameters
```bash
./display --list-cameras
```
Output:
```
=== Camera List ===
Found 2 camera(s):

Camera 0:
  ID: 0
  Name: Video Device 0
  Path: /dev/video0
  Driver: v4l2
  Card: Camera 0
  Bus Info: platform
  Capabilities: [VIDEO_CAPTURE]
  Supported Formats (3):
    [0] BGR - BGR 24-bit (640x480)
    [1] BGR - BGR 24-bit (1280x720)
    [2] BGR - BGR 24-bit (1920x1080)
  Controls (19):
    [0] brightness (Brightness)
        Type: integer
        Range: 0 - 255 (default: 128, step: 1)
    [1] contrast (Contrast)
        Type: integer
        Range: 0 - 255 (default: 128, step: 1)
    ... (more controls)

Camera 1:
  ID: 1
  Name: Video Device 1
  Path: /dev/video1
  ...
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