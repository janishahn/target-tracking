#!/usr/bin/env python3
"""
Raspberry Pi Zero 2W Target Tracking with Live HTTP MJPEG Streaming and Servo Control
Streams annotated video feed to network while maintaining headless operation
Controls 4 servos based on target position relative to frame center
Access stream at: http://PI_IP_ADDRESS:8080/stream
"""

import time
import threading
import argparse
import os
from queue import Queue
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
from picamera2 import Picamera2
from target_tracker import TargetTracker, Config

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Servo control will be simulated.")
    GPIO_AVAILABLE = False

class ServoController:
    """
    Controls 4 servos based on target position with robust actuation and minimum deflection thresholds.
    Servo layout: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    """
    
    def __init__(self, pins=[18, 19, 20, 21], min_deflection=2.0, pwm_frequency=50):
        """
        Initialize servo controller.
        
        Args:
            pins: GPIO pins for servos [top_left, top_right, bottom_left, bottom_right]
            min_deflection: Minimum angle change required to update servo (degrees)
            pwm_frequency: PWM frequency in Hz (typically 50Hz for servos)
        """
        self.pins = pins
        self.min_deflection = min_deflection
        self.pwm_frequency = pwm_frequency
        self.current_angles = [90.0, 90.0, 90.0, 90.0]  # Start at center position
        self.pwm_objects = []
        
        # Frame dimensions for offset calculations
        self.frame_width = Config.FRAME_WIDTH
        self.frame_height = Config.FRAME_HEIGHT
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Servo angle limits
        self.min_angle = 45.0
        self.max_angle = 135.0
        self.center_angle = 90.0
        
        self.initialized = False
        self.initialize_servos()
    
    def initialize_servos(self):
        """Initialize GPIO and PWM for servo control."""
        if not GPIO_AVAILABLE:
            print("Servo control simulation mode (RPi.GPIO not available)")
            self.initialized = True
            return
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            for pin in self.pins:
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, self.pwm_frequency)
                pwm.start(self.angle_to_duty_cycle(self.center_angle))
                self.pwm_objects.append(pwm)
            
            print(f"Servos initialized on pins {self.pins}")
            print(f"Servo range: {self.min_angle}° - {self.max_angle}°")
            print(f"Minimum deflection threshold: {self.min_deflection}°")
            self.initialized = True
            
        except Exception as e:
            print(f"Failed to initialize servos: {e}")
            self.initialized = False
    
    def angle_to_duty_cycle(self, angle):
        """
        Convert servo angle to PWM duty cycle.
        
        Args:
            angle: Servo angle in degrees (0-180)
            
        Returns:
            PWM duty cycle percentage (typically 2.5-12.5 for 0-180 degrees)
        """
        # Standard servo: 1ms pulse = 0°, 1.5ms pulse = 90°, 2ms pulse = 180°
        # For 50Hz PWM: 1ms = 5% duty cycle, 1.5ms = 7.5%, 2ms = 10%
        min_duty = 2.5  # 0.5ms pulse width
        max_duty = 12.5  # 2.5ms pulse width
        duty_cycle = min_duty + (angle / 180.0) * (max_duty - min_duty)
        return duty_cycle
    
    def calculate_servo_angles(self, x_offset, y_offset):
        """
        Calculate servo angles based on target offset from frame center.
        
        Args:
            x_offset: Horizontal offset from center (positive = right)
            y_offset: Vertical offset from center (positive = down)
            
        Returns:
            List of servo angles [top_left, top_right, bottom_left, bottom_right]
        """
        # Normalize offsets to [-1, 1] range
        max_x_offset = self.frame_center_x
        max_y_offset = self.frame_center_y
        
        norm_x = max(-1.0, min(1.0, x_offset / max_x_offset))
        norm_y = max(-1.0, min(1.0, y_offset / max_y_offset))
        
        # Calculate angle range (45° range around center)
        angle_range = (self.max_angle - self.min_angle) / 2.0  # 45°
        
        # Calculate individual servo angles based on quadrant influence
        # Top-left servo: influenced by negative x and negative y
        top_left = self.center_angle + angle_range * (-norm_x - norm_y) / 2.0
        
        # Top-right servo: influenced by positive x and negative y  
        top_right = self.center_angle + angle_range * (norm_x - norm_y) / 2.0
        
        # Bottom-left servo: influenced by negative x and positive y
        bottom_left = self.center_angle + angle_range * (-norm_x + norm_y) / 2.0
        
        # Bottom-right servo: influenced by positive x and positive y
        bottom_right = self.center_angle + angle_range * (norm_x + norm_y) / 2.0
        
        # Clamp angles to valid range
        angles = [
            max(self.min_angle, min(self.max_angle, top_left)),
            max(self.min_angle, min(self.max_angle, top_right)),
            max(self.min_angle, min(self.max_angle, bottom_left)),
            max(self.min_angle, min(self.max_angle, bottom_right))
        ]
        
        return angles
    
    def update_servos(self, x_offset, y_offset, has_target=True):
        """
        Update servo positions based on target offset with minimum deflection filtering.
        
        Args:
            x_offset: Horizontal offset from center
            y_offset: Vertical offset from center  
            has_target: Whether a target is currently detected
        """
        if not self.initialized:
            return
        
        if not has_target:
            # Return to center position when no target
            target_angles = [self.center_angle] * 4
        else:
            # Calculate target angles based on offset
            target_angles = self.calculate_servo_angles(x_offset, y_offset)
        
        # Apply minimum deflection filtering
        for i, target_angle in enumerate(target_angles):
            angle_diff = abs(target_angle - self.current_angles[i])
            
            if angle_diff >= self.min_deflection:
                self.current_angles[i] = target_angle
                
                if GPIO_AVAILABLE and i < len(self.pwm_objects):
                    duty_cycle = self.angle_to_duty_cycle(target_angle)
                    self.pwm_objects[i].ChangeDutyCycle(duty_cycle)
    
    def get_servo_status(self):
        """
        Get current servo status for debugging.
        
        Returns:
            Dictionary with servo angles and status
        """
        return {
            'angles': self.current_angles.copy(),
            'pins': self.pins,
            'initialized': self.initialized,
            'gpio_available': GPIO_AVAILABLE
        }
    
    def cleanup(self):
        """Clean up GPIO resources."""
        if GPIO_AVAILABLE and self.initialized:
            for pwm in self.pwm_objects:
                pwm.stop()
            GPIO.cleanup()
            print("Servo GPIO cleanup completed")

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming"""
    
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            
            try:
                while True:
                    if hasattr(self.server, 'current_frame') and self.server.current_frame is not None:
                        # Encode frame as JPEG
                        ret, jpeg = cv2.imencode('.jpg', self.server.current_frame, 
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            self.wfile.write(b'--jpgboundary\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                    
                    time.sleep(0.033)  # ~30 FPS max
                    
            except Exception as e:
                print(f"Streaming error: {e}")
                
        elif self.path == '/':
            # Serve a simple HTML page to view the stream
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pi Zero Target Tracker Stream</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                    .stream {{ text-align: center; margin: 20px 0; }}
                    .info {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    img {{ max-width: 100%; border: 2px solid #333; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎯 Pi Zero Target Tracker Live Stream</h1>
                    <div class="info">
                        <strong>Status:</strong> Tracking GREEN objects with servo control<br>
                        <strong>Resolution:</strong> 640x480 (processed from 1280x960 capture)<br>
                        <strong>Device:</strong> Raspberry Pi Zero 2W<br>
                        <strong>Servos:</strong> 4x servos (45-135° range, center at 90°)
                    </div>
                    <div class="stream">
                        <img src="/stream" alt="Live Stream">
                    </div>
                    <div class="info">
                        <strong>Stream URL:</strong> http://{self.get_pi_ip()}:8080/stream<br>
                        <strong>Access:</strong> Open this URL in VLC, browser, or any MJPEG viewer
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_error(404)
    
    def get_pi_ip(self):
        """Get Pi's IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "PI_IP_ADDRESS"
    
    def log_message(self, format, *args):
        # Suppress HTTP log messages
        pass

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threading HTTP server for concurrent connections"""
    allow_reuse_address = True
    daemon_threads = True



def get_pi_ip_address():
    """Get the Pi's IP address for network access"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to determine IP"

def start_streaming_server(port=8080):
    """Start the HTTP streaming server"""
    try:
        server = ThreadingHTTPServer(('0.0.0.0', port), StreamingHandler)
        server.current_frame = None
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        pi_ip = get_pi_ip_address()
        print(f"🌐 Streaming server started!")
        print(f"📺 View stream at: http://{pi_ip}:{port}")
        print(f"🔗 Direct stream URL: http://{pi_ip}:{port}/stream")
        print(f"💻 Open in browser, VLC, or any MJPEG viewer")
        
        return server
        
    except Exception as e:
        print(f"Failed to start streaming server: {e}")
        return None

def initialize_camera_pi_zero() -> Picamera2:
    """
    Initialize camera optimized for Pi Zero 2W headless operation with larger FOV
    Captures at double resolution (640x480) then downsamples to maintain processing workload
    """
    print("Initializing Raspberry Pi camera (headless mode with larger FOV)...")
    picam2 = Picamera2()
    
    # Configure camera to capture at double resolution for larger field of view
    # We'll downsample to 640x480 for processing to maintain Pi Zero workload
    camera_config = picam2.create_video_configuration(
        main={"size": (1280, 960), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(camera_config)
    picam2.start()
    
    print("Camera initialized successfully (1280x960 capture -> 640x480 processing)")
    return picam2

def camera_capture_thread(picam2: Picamera2, frame_queue: Queue):
    """
    Optimized capture thread for headless operation with downsampling
    Captures at 1280x960 and downsamples to 640x480 for processing
    """
    print("Starting camera capture thread (with downsampling)...")
    while True:
        try:
            # Capture at full resolution (1280x960)
            frame_rgb_full = picam2.capture_array("main")
            
            # Downsample to target processing resolution (640x480)
            # This maintains the same processing workload while providing larger FOV
            frame_rgb_downsampled = cv2.resize(frame_rgb_full, (640, 480), interpolation=cv2.INTER_AREA)
            
            if not frame_queue.full():
                frame_queue.put_nowait(frame_rgb_downsampled)
                
        except Exception as e:
            print(f"Error in capture thread: {e}")
            break

def video_file_thread(video_path: str, frame_queue: Queue):
    """
    Video file reading thread for MP4 processing
    """
    print(f"Starting video file thread for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0  # Default to 30 FPS if unknown
    
    print(f"Video FPS: {fps:.1f}, Frame delay: {frame_delay:.3f}s")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached")
            break
            
        try:
            # Resize frame to match processing resolution (640x480)
            frame_resized = cv2.resize(frame, (640, 480))
            
            if not frame_queue.full():
                frame_queue.put_nowait(frame_resized)
            
            # Maintain original video timing
            time.sleep(frame_delay)
                
        except Exception as e:
            print(f"Error in video file thread: {e}")
            break
    
    cap.release()
    print("Video file processing completed")



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Raspberry Pi Zero 2W Target Tracker with Live Streaming and Servo Control')
    parser.add_argument('--source', type=str, help='MP4 video file to process instead of live camera')
    parser.add_argument('--port', type=int, default=8080, help='HTTP streaming port (default: 8080)')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming (headless only)')
    parser.add_argument('--no-servos', action='store_true', help='Disable servo control')
    parser.add_argument('--servo-pins', nargs=4, type=int, default=[18, 19, 20, 21], 
                        help='GPIO pins for servos [top_left, top_right, bottom_left, bottom_right] (default: 18 19 20 21)')
    parser.add_argument('--min-deflection', type=float, default=2.0, 
                        help='Minimum servo angle change to prevent twitching (default: 2.0 degrees)')
    return parser.parse_args()

def main():
    """
    Main application loop for Pi Zero 2W with live streaming
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("RASPBERRY PI ZERO 2W TARGET TRACKER WITH LIVE STREAMING AND SERVO CONTROL")
    print("=" * 60)
    
    if args.source:
        print(f"Processing video file: {args.source}")
        if not os.path.exists(args.source):
            print(f"Error: Video file '{args.source}' not found!")
            return
    else:
        print("Using live camera feed...")
    
    if not args.no_stream:
        print(f"Live streaming enabled on port {args.port}")
    else:
        print("Streaming disabled - headless mode only")
    
    if not args.no_servos:
        print(f"Servo control enabled on pins {args.servo_pins}")
        print(f"Minimum deflection threshold: {args.min_deflection}°")
    else:
        print("Servo control disabled")
    
    # Configuration for streaming operation
    status_interval = 150   # Print status every N frames
    
    # Enable diagnostic mode for detection debugging
    diagnostic_mode = True
    
    # Disable all GUI-related config options
    Config.SHOW_FPS = False
    Config.SHOW_MASK = False
    Config.DEBUG = False
    Config.ENABLE_RESULT_LOGGING = False  # Disable the built-in logging
    
    # Configure for GREEN object tracking (default ranges)
    Config.HSV_LOWER1 = (45, 60, 60)    # Green range (conservative)
    Config.HSV_UPPER1 = (75, 255, 255)
    Config.HSV_LOWER2 = (40, 50, 50)    # Extended green range
    Config.HSV_UPPER2 = (80, 255, 255)
    
    # Allow larger green blobs (increase max detection radius for close objects)
    Config.MAX_DETECTION_RADIUS = 200  # Increased from 100 to 200 pixels
    
    # Diagnostic: Relax detection parameters for better detection
    if diagnostic_mode:
        print("DIAGNOSTIC MODE ENABLED - Relaxing detection parameters for GREEN objects")
        Config.MIN_AREA = 50  # Reduce from 200 to 50
        Config.MIN_CIRCULARITY = 0.3  # Reduce from 0.6 to 0.3
        Config.MAX_DETECTION_RADIUS = 250  # Allow very large green blobs when close to camera
        # Green HSV ranges - green doesn't wrap around like red, so we use one main range
        Config.HSV_LOWER1 = (40, 50, 50)    # Green range (hue 40-80)
        Config.HSV_UPPER1 = (80, 255, 255)
        Config.HSV_LOWER2 = (35, 40, 40)    # Extended green range for edge cases
        Config.HSV_UPPER2 = (85, 255, 255)
    
    print(f"Status interval: every {status_interval} frames")
    
    # Step 1: Start streaming server if enabled
    streaming_server = None
    if not args.no_stream:
        streaming_server = start_streaming_server(args.port)
        if streaming_server is None:
            print("Warning: Streaming server failed to start, continuing in headless mode")
    
    # Step 2: Initialize video source
    picam2 = None
    if args.source:
        # Video file mode - no camera initialization needed
        print("Video file mode - skipping camera initialization")
    else:
        # Live camera mode
        try:
            picam2 = initialize_camera_pi_zero()
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            print("Make sure the camera is properly connected and enabled.")
            return
    
    # Step 3: Setup frame queue and capture thread
    frame_queue = Queue(maxsize=2)
    
    if args.source:
        # Start video file thread
        capture_thread = threading.Thread(
            target=video_file_thread, 
            args=(args.source, frame_queue), 
            daemon=True
        )
    else:
        # Start camera capture thread
        capture_thread = threading.Thread(
            target=camera_capture_thread, 
            args=(picam2, frame_queue), 
            daemon=True
        )
    
    capture_thread.start()
    
    # Step 4: Initialize servo controller
    servo_controller = None
    if not args.no_servos:
        try:
            servo_controller = ServoController(
                pins=args.servo_pins,
                min_deflection=args.min_deflection
            )
            print("Servo controller initialized successfully")
        except Exception as e:
            print(f"Failed to initialize servo controller: {e}")
            servo_controller = None
    
    # Step 5: Initialize tracking components
    video_source = args.source if args.source else 0  # 0 for camera, file path for video
    tracker = TargetTracker(result_logger=None, video_source=video_source)  # No result logger
    
    print("\nTarget tracking with live streaming and servo control started!")
    print(f"Camera FOV: 1280x960 (double resolution for larger field of view)")
    print(f"Processing resolution: 640x480 (maintains Pi Zero workload)")
    print(f"Target: Green objects, min area: {Config.MIN_AREA} pixels")
    if servo_controller:
        servo_status = servo_controller.get_servo_status()
        print(f"Servo control: 4 servos on pins {servo_status['pins']}")
        print(f"Servo range: 45-135°, center: 90°, min deflection: {args.min_deflection}°")
    if diagnostic_mode:
        print(f"DIAGNOSTIC MODE: Relaxed parameters - Area≥{Config.MIN_AREA}, Circularity≥{Config.MIN_CIRCULARITY}")
        print(f"HSV ranges: {Config.HSV_LOWER1}-{Config.HSV_UPPER1} and {Config.HSV_LOWER2}-{Config.HSV_UPPER2}")
    
    if streaming_server:
        pi_ip = get_pi_ip_address()
        print(f"\n🌐 LIVE STREAM ACCESS:")
        print(f"📱 Browser: http://{pi_ip}:{args.port}")
        print(f"🎥 VLC: http://{pi_ip}:{args.port}/stream")
        print(f"📺 Direct URL: http://{pi_ip}:{args.port}/stream")
    
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Step 5: Main processing loop
        while True:
            # Get frame from capture queue
            if frame_queue.empty():
                # For video files, check if thread is still alive
                if args.source and not capture_thread.is_alive():
                    print("Video file processing completed")
                    break
                time.sleep(0.001)
                continue
                
            frame_rgb = frame_queue.get()
            
            # Convert frame format if needed
            if args.source:
                # Video file frames are already in BGR format
                frame_bgr = frame_rgb
            else:
                # Camera frames are in RGB format, convert to BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Process frame through tracker pipeline
            annotated_frame = tracker.process_frame(frame_bgr)
            
            # Update streaming server with latest frame
            if streaming_server:
                streaming_server.current_frame = annotated_frame.copy()
            
            # Get tracking state with distance information
            cx, cy, radius, locked, bbox, distance_info = tracker.get_state_with_distance()
            abs_distance, x_offset, y_offset, frame_center_x, frame_center_y = distance_info
            has_detection = (cx != -1 and cy != -1)
            
            # Update servo positions based on target offset
            if servo_controller:
                servo_controller.update_servos(x_offset, y_offset, has_detection and locked)
            
            # Print status periodically
            if tracker.frame_count % status_interval == 0:
                print(f"\n--- Frame {tracker.frame_count} Status ---")
                if locked:
                    print(f"Target: ({cx}, {cy}) radius={radius}")
                    direction_desc = tracker.get_directional_description(x_offset, y_offset)
                    print(f"Distance: {abs_distance:.1f}px from center - {direction_desc}")
                    print(f"Offset: ({x_offset:+d}, {y_offset:+d})")
                else:
                    print("Status: TARGET LOST")
                
                # Show servo status
                if servo_controller:
                    servo_status = servo_controller.get_servo_status()
                    angles = servo_status['angles']
                    print(f"Servos: TL={angles[0]:.1f}° TR={angles[1]:.1f}° BL={angles[2]:.1f}° BR={angles[3]:.1f}°")
                
                if streaming_server:
                    pi_ip = get_pi_ip_address()
                    print(f"Stream: http://{pi_ip}:{args.port}")
                print("-" * 35)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Step 6: Cleanup and final reporting
        print("Shutting down tracker, streaming, and servos...")
        
        print(f"\nTotal frames processed: {tracker.frame_count}")
        
        # Stop camera if it was used
        if picam2:
            picam2.stop()
            print("Camera stopped")
        
        # Stop streaming server
        if streaming_server:
            streaming_server.shutdown()
            print("Streaming server stopped")
        
        # Cleanup servo controller
        if servo_controller:
            servo_controller.cleanup()
        
        print("Target tracker with streaming and servo control stopped successfully!")

if __name__ == "__main__":
    main() 