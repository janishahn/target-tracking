#!/usr/bin/env python3
"""
Raspberry Pi Zero 2W Target Tracking with Live HTTP MJPEG Streaming
Streams annotated video feed to network while maintaining headless operation
Access stream at: http://PI_IP_ADDRESS:8080/stream
"""

import time
import threading
import argparse
from queue import Queue
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
from picamera2 import Picamera2
from target_tracker import TargetTracker, Config

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
                        <strong>Status:</strong> Tracking GREEN objects<br>
                        <strong>Resolution:</strong> 320x240 (processed from 640x480 capture)<br>
                        <strong>Device:</strong> Raspberry Pi Zero 2W
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
    # We'll downsample to 320x240 for processing to maintain Pi Zero workload
    camera_config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(camera_config)
    picam2.start()
    
    print("Camera initialized successfully (640x480 capture -> 320x240 processing)")
    return picam2

def camera_capture_thread(picam2: Picamera2, frame_queue: Queue):
    """
    Optimized capture thread for headless operation with downsampling
    Captures at 640x480 and downsamples to 320x240 for processing
    """
    print("Starting camera capture thread (with downsampling)...")
    while True:
        try:
            # Capture at full resolution (640x480)
            frame_rgb_full = picam2.capture_array("main")
            
            # Downsample to target processing resolution (320x240)
            # This maintains the same processing workload while providing larger FOV
            frame_rgb_downsampled = cv2.resize(frame_rgb_full, (320, 240), interpolation=cv2.INTER_AREA)
            
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
            # Resize frame to match processing resolution (320x240)
            frame_resized = cv2.resize(frame, (320, 240))
            
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
    parser = argparse.ArgumentParser(description='Raspberry Pi Zero 2W Target Tracker with Live Streaming')
    parser.add_argument('--source', type=str, help='MP4 video file to process instead of live camera')
    parser.add_argument('--port', type=int, default=8080, help='HTTP streaming port (default: 8080)')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming (headless only)')
    return parser.parse_args()

def main():
    """
    Main application loop for Pi Zero 2W with live streaming
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("RASPBERRY PI ZERO 2W TARGET TRACKER WITH LIVE STREAMING")
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
    
    # Step 4: Initialize tracking components
    video_source = args.source if args.source else 0  # 0 for camera, file path for video
    tracker = TargetTracker(result_logger=None, video_source=video_source)  # No result logger
    
    print("\nTarget tracking with live streaming started!")
    print(f"Camera FOV: 640x480 (double resolution for larger field of view)")
    print(f"Processing resolution: 320x240 (maintains Pi Zero workload)")
    print(f"Target: Green objects, min area: {Config.MIN_AREA} pixels")
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
            
            # Print status periodically
            if tracker.frame_count % status_interval == 0:
                print(f"\n--- Frame {tracker.frame_count} Status ---")
                if locked:
                    print(f"Target: ({cx}, {cy}) radius={radius}")
                    direction_desc = tracker.get_directional_description(x_offset, y_offset)
                    print(f"Distance: {abs_distance:.1f}px from center - {direction_desc}")
                else:
                    print("Status: TARGET LOST")
                if streaming_server:
                    pi_ip = get_pi_ip_address()
                    print(f"Stream: http://{pi_ip}:{args.port}")
                print("-" * 35)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Step 6: Cleanup and final reporting
        print("Shutting down tracker and streaming...")
        
        print(f"\nTotal frames processed: {tracker.frame_count}")
        
        # Stop camera if it was used
        if picam2:
            picam2.stop()
            print("Camera stopped")
        
        # Stop streaming server
        if streaming_server:
            streaming_server.shutdown()
            print("Streaming server stopped")
        
        print("Target tracker with streaming stopped successfully!")

if __name__ == "__main__":
    main() 