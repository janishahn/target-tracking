# Ultra-Lightweight Target Detection & Tracking Pipeline

A single-file target tracking system designed for real-time detection and tracking of colored objects (specifically red circles) using only OpenCV. Optimized for both MacBook development and Raspberry Pi Zero W deployment without code changes.

## Features

- **Ultra-lightweight**: Only depends on OpenCV and NumPy
- **Single-file monolith**: Everything in one Python file for easy deployment
- **Cross-platform**: Works on macOS, Linux, and Raspberry Pi without modifications
- **Real-time performance**: Optimized for 15-20 FPS on Raspberry Pi Zero W
- **Color-based detection**: HSV color thresholding for robust target detection
- **Temporal tracking**: Exponential moving average smoothing and persistence
- **Visual feedback**: Real-time visualization with crosshairs and status overlay
- **Servo control**: 4-servo actuation system with robust anti-twitching algorithms
- **Live streaming**: HTTP MJPEG streaming for remote monitoring

## Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- RPi.GPIO (for servo control on Raspberry Pi)
- picamera2 (for Pi camera support)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run with a demo video file:
```bash
python target_tracker.py --source demo.mp4
```

Run with camera (webcam or USB camera):
```bash
python target_tracker.py --camera
```

### NEW: Live Camera Streaming

For live video streaming with camera:

**macOS Built-in Webcam (Development):**
```bash
python3 main_macos.py
```

**Raspberry Pi Camera (Standard):**
```bash
python3 main_pi.py
```

**Raspberry Pi Camera (Pi Zero 2W Optimized):**
```bash
python3 main_pi_zero.py
```

**Pi Zero 2W with Live Streaming and Servo Control:**
```bash
python3 main_pi_zero_streaming.py
```

**Headless operation (SSH/remote):**
```bash
python3 main_pi_headless.py
```

> **📖 See [PI_CAMERA_SETUP.md](PI_CAMERA_SETUP.md) for complete setup guide**

### Command Line Options

- `--source <path>`: Specify video file or camera index
- `--camera`: Use camera instead of video file (equivalent to `--source 0`)
- `--debug`: Enable debug mode with mask visualization
- `--csv`: Output tracking data in CSV format
- `--no-display`: Run headless without GUI (useful for Pi deployment)

### Examples

**Development on MacBook with demo video:**
```bash
python target_tracker.py --source demo.mp4 --debug
```

**Production on Raspberry Pi with USB camera:**
```bash
USE_PI_CAMERA=1 python target_tracker.py --camera --no-display --csv
```

**Live camera streaming:**
```bash
python3 main_macos.py           # macOS built-in webcam
python3 main_pi.py              # Raspberry Pi camera (full GUI)
python3 main_pi_headless.py     # Raspberry Pi camera (headless)
```

**Pi Zero with servo control and streaming:**
```bash
python3 main_pi_zero_streaming.py                    # Full servo control
python3 main_pi_zero_streaming.py --no-servos        # Disable servos
python3 main_pi_zero_streaming.py --servo-pins 12 13 16 18  # Custom pins
python3 main_pi_zero_streaming.py --min-deflection 3.0      # Adjust sensitivity
```

**CSV output for data analysis:**
```bash
python target_tracker.py --source demo.mp4 --csv > tracking_data.csv
```

### Interactive Controls

When running with display:
- `q`: Quit the application
- `s`: Save current frame as image
- `m`: Toggle color mask visualization
- `b`: Toggle bounding box display
- `c`: Toggle color format (RGB/BGR) - fixes blue vs red tracking issues
- `d`: Toggle debug mode

## Configuration

All parameters can be adjusted in the `Config` class within `target_tracker.py`:

### Target Detection Parameters
- `HSV_LOWER1/UPPER1`: Lower red HSV range (0-10°)
- `HSV_LOWER2/UPPER2`: Upper red HSV range (160-180°)
- `MIN_AREA`: Minimum contour area in pixels (default: 200)
- `MIN_CIRCULARITY`: Minimum circularity 0-1 (default: 0.6)
- `MAX_DETECTION_RADIUS`: Maximum detection radius (default: 100)

### Tracking Parameters
- `SMOOTHING_ALPHA`: EMA smoothing factor (default: 0.3)
- `MAX_MISSING_FRAMES`: Frames before declaring target lost (default: 5)
- `SEARCH_WINDOW_SIZE`: Tracking window radius (default: 20)

### Performance Parameters
- `FRAME_WIDTH/HEIGHT`: Processing resolution (default: 640x480)

## Servo Control System

The Pi Zero streaming version includes a 4-servo control system that responds to target position:

### Hardware Setup
- **4 servos** connected to GPIO pins (default: 18, 19, 20, 21)
- **Servo layout**: Top-Left, Top-Right, Bottom-Left, Bottom-Right
- **Power**: Ensure adequate 5V power supply for servos
- **Wiring**: Connect servo signal wires to GPIO pins, power to 5V rail

### Servo Behavior
- **Center position**: All servos at 90° when target is at frame center (320, 240)
- **Angle range**: 45-135° (90° ± 45°) for maximum deflection at frame edges
- **Anti-twitching**: Minimum deflection threshold (default: 2°) prevents small movements
- **Smooth response**: Servos only update when angle change exceeds threshold

### Servo Control Options
```bash
# Test servo functionality
python3 test_servos.py                    # Test all servo movements
python3 test_servos.py --test center      # Test center position only
python3 test_servos.py --pins 12 13 16 18 # Use custom GPIO pins

# Main application with servo control
python3 main_pi_zero_streaming.py --servo-pins 18 19 20 21  # Default pins
python3 main_pi_zero_streaming.py --min-deflection 3.0     # Reduce sensitivity
python3 main_pi_zero_streaming.py --no-servos              # Disable servos
```

### Servo Angle Calculation
The system maps target position to servo angles using quadrant influence:
- **Top-Left servo**: Responds to negative X and negative Y offsets
- **Top-Right servo**: Responds to positive X and negative Y offsets  
- **Bottom-Left servo**: Responds to negative X and positive Y offsets
- **Bottom-Right servo**: Responds to positive X and positive Y offsets

Target at frame center (0,0 offset) → All servos at 90°
Target at top-left corner → TL servo at 135°, others compensate
Target at bottom-right corner → BR servo at 135°, others compensate

## Target Specifications

The system is optimized to detect:
- **Red circular objects** (painted red circles, red balls, etc.)
- **Minimum size**: 200 pixels area (roughly 16x16 pixel square)
- **Shape**: Circular objects with circularity ≥ 0.6
- **Color**: Red in HSV space (handles lighting variations better than RGB)

## Output Format

When using `--csv` flag, outputs tracking data in CSV format:
```
frame,cx,cy,radius,locked
1,160.50,120.25,15.2,1
2,161.20,121.10,15.8,1
3,0,0,0,0
```

Where:
- `frame`: Frame number
- `cx,cy`: Target centroid coordinates (pixels)
- `radius`: Target radius (pixels)
- `locked`: 1 if target is locked, 0 if lost

## Raspberry Pi Deployment

### Setup on Raspberry Pi Zero W

1. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-opencv python3-numpy
   ```

2. Copy the single file:
   ```bash
   scp target_tracker.py pi@your-pi-ip:~/
   ```

3. Run with Pi camera:
   ```bash
   USE_PI_CAMERA=1 python3 target_tracker.py --camera --no-display --csv
   ```

### Performance Optimization for Pi

The system is pre-configured for Pi performance:
- Medium resolution (640x480) for balanced processing
- Minimal morphological operations
- Efficient contour filtering
- Optimized for 15-20 FPS on Pi Zero W

## Troubleshooting

### Common Issues

**"Could not open video source"**
- Check if camera is connected and not used by another application
- Try different camera indices (0, 1, 2...)
- Ensure video file path is correct

**Tracking blue objects instead of red objects**
- This is a camera color format issue (RGB vs BGR)
- The system now auto-detects and fixes this automatically
- If auto-detection fails, press 'c' during tracking to toggle color format
- For permanent fix, set `Config.FORCE_COLOR_FORMAT = True` (for RGB cameras) or `False` (for BGR cameras)
- Run `python test_color_fix.py` to diagnose and test the fix

**Poor detection performance**
- Adjust HSV thresholds in Config class
- Ensure good lighting conditions
- Check target size meets minimum area requirement

**Low FPS on Raspberry Pi**
- Reduce frame resolution in Config
- Disable debug visualization
- Use `--no-display` flag

**Servo control issues**
- Check GPIO permissions: `sudo usermod -a -G gpio $USER` (logout/login required)
- Verify servo power supply (5V, adequate current)
- Test servos individually: `python3 test_servos.py --test center`
- Check GPIO pin conflicts with other devices
- Ensure RPi.GPIO is installed: `pip install RPi.GPIO`

**Servos twitching or jittery**
- Increase minimum deflection threshold: `--min-deflection 3.0`
- Check power supply stability
- Verify servo signal wire connections
- Reduce tracking sensitivity in Config class

### HSV Color Tuning

To tune HSV values for your specific target:

1. Run with debug mode: `python target_tracker.py --debug`
2. Press `m` to show color mask
3. Adjust `HSV_LOWER1/UPPER1` and `HSV_LOWER2/UPPER2` in Config class
4. Red color wraps around HSV hue, so two ranges are needed

## Architecture

The system follows a simple pipeline:

1. **Frame Acquisition**: Capture from video/camera and resize
2. **Color Filtering**: HSV thresholding to create binary mask
3. **Contour Detection**: Find and filter contours by area/circularity
4. **Target Selection**: Choose best candidate contour
5. **Tracking**: Temporal smoothing and persistence
6. **Visualization**: Overlay tracking information
7. **Output**: CSV data or visual display

## License

This project is designed for educational and research purposes. Feel free to modify and adapt for your specific use case.

## Contributing

This is a single-file system designed for simplicity. For modifications:
1. All changes should maintain the single-file architecture
2. Ensure compatibility with both development (macOS) and deployment (Pi) environments
3. Test performance on target hardware (Pi Zero W) 