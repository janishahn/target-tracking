#!/usr/bin/env python3
"""
Headless Raspberry Pi Camera Integration for Target Tracking
No GUI display - for performance testing and remote deployment
"""

import time
import threading
import signal
import sys
from queue import Queue

import cv2
from picamera2 import Picamera2
from target_tracker import TargetTracker, ResultLogger, Config

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    print("\nShutdown signal received...")
    shutdown_flag = True

def initialize_camera_headless() -> Picamera2:
    """
    Initialize camera for headless operation
    """
    print("Initializing Raspberry Pi camera (headless mode)...")
    picam2 = Picamera2()
    
    # Configure camera
    camera_config = picam2.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(camera_config)
    
    # Start the camera
    picam2.start()
    print("Camera initialized successfully (headless mode)")
    return picam2

def camera_capture_thread_headless(picam2: Picamera2, frame_queue: Queue):
    """
    Headless capture thread
    """
    global shutdown_flag
    print("Starting headless camera capture thread...")
    
    while not shutdown_flag:
        try:
            # Capture frame as RGB888 array
            frame_rgb = picam2.capture_array("main")
            
            # Non-blocking put
            if not frame_queue.full():
                frame_queue.put_nowait(frame_rgb)
                
        except Exception as e:
            if not shutdown_flag:
                print(f"Error in capture thread: {e}")
            break

def main():
    """
    Main headless application loop
    """
    global shutdown_flag
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("RASPBERRY PI HEADLESS TARGET TRACKER")
    print("=" * 60)
    print("Headless mode - no GUI display")
    print("Press Ctrl+C to stop...")
    
    # Override config for headless operation
    Config.SHOW_FPS = False
    Config.SHOW_MASK = False
    Config.DEBUG = False  # Reduce debug output
    
    # Step 1: Initialize camera
    try:
        picam2 = initialize_camera_headless()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return 1
    
    # Step 2: Setup frame capture
    frame_queue = Queue(maxsize=1)
    capture_thread = threading.Thread(
        target=camera_capture_thread_headless, 
        args=(picam2, frame_queue), 
        daemon=True
    )
    capture_thread.start()
    
    # Step 3: Initialize tracking components
    result_logger = ResultLogger("PiCamera_Headless") if Config.ENABLE_RESULT_LOGGING else None
    tracker = TargetTracker(result_logger)
    
    # Step 4: Print status
    print(f"Tracking started - Resolution: 320x240")
    print(f"Target: Red objects, min area: {Config.MIN_AREA} pixels")
    print(f"Result logging: {'ON' if Config.ENABLE_RESULT_LOGGING else 'OFF'}")
    print("Running headless tracking loop...")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_status_time = time.time()
    last_detection_time = None
    detection_count = 0
    
    try:
        # Step 5: Main processing loop
        while not shutdown_flag:
            # Get frame from capture queue
            if frame_queue.empty():
                time.sleep(0.001)
                continue
                
            frame_rgb = frame_queue.get()
            
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Process frame through tracker pipeline
            _ = tracker.process_frame(frame_bgr)
            
            # Get tracking state
            cx, cy, radius, locked, bbox = tracker.get_state()
            
            # Count detections
            if cx != -1 and cy != -1:
                detection_count += 1
                last_detection_time = time.time()
            
            frame_count += 1
            
            # Print status every 5 seconds
            current_time = time.time()
            if current_time - last_status_time >= 5.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
                
                print(f"Status: Frame {frame_count}, FPS: {fps:.1f}, "
                      f"Detections: {detection_rate:.1f}%, "
                      f"Locked: {'YES' if locked else 'NO'}")
                
                if last_detection_time:
                    time_since_detection = current_time - last_detection_time
                    if time_since_detection < 1.0:
                        print(f"  Last detection: {time_since_detection:.1f}s ago at ({cx}, {cy})")
                
                last_status_time = current_time
    
    except Exception as e:
        print(f"Error in main loop: {e}")
        return 1
    
    finally:
        # Step 6: Cleanup
        print("\nShutting down headless tracker...")
        
        # Final statistics
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        final_detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
        
        print(f"Final Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {final_fps:.1f}")
        print(f"  Detection rate: {final_detection_rate:.1f}%")
        print(f"  Total detections: {detection_count}")
        
        # Finalize logging session
        if result_logger:
            result_logger.finalize_session(final_fps)
        
        # Stop camera
        picam2.stop()
        print("Camera stopped")
        
        print("Headless tracker shutdown complete.")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 