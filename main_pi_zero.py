#!/usr/bin/env python3
"""
Raspberry Pi Zero 2W Optimized Camera Integration for Target Tracking
Performance-optimized version for Pi Zero 2W with lower resolution and reduced overhead
"""

import time
import threading
from queue import Queue

import cv2
from picamera2 import Picamera2
from target_tracker import TargetTracker, ResultLogger, Config

def initialize_camera_pi_zero() -> Picamera2:
    """
    Initialize camera optimized for Pi Zero 2W performance
    Uses lower resolution for better framerate
    """
    print("Initializing Raspberry Pi camera (Pi Zero optimized)...")
    picam2 = Picamera2()
    
    # Configure camera for optimal Pi Zero performance
    # Using 320x240 to match tracker resolution exactly (no resize needed)
    camera_config = picam2.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        # Lower buffer count for reduced memory usage
        buffer_count=2
    )
    picam2.configure(camera_config)
    
    # Start the camera
    picam2.start()
    print("Camera initialized successfully (320x240 @ low latency)")
    return picam2

def camera_capture_thread_optimized(picam2: Picamera2, frame_queue: Queue):
    """
    Optimized capture thread for Pi Zero with minimal overhead
    """
    print("Starting optimized camera capture thread...")
    while True:
        try:
            # Capture frame as RGB888 array
            frame_rgb = picam2.capture_array("main")
            
            # Non-blocking put with immediate drop if queue full
            if not frame_queue.full():
                frame_queue.put_nowait(frame_rgb)
                
        except Exception as e:
            print(f"Error in capture thread: {e}")
            break

def main():
    """
    Main application loop optimized for Pi Zero 2W
    """
    print("=" * 60)
    print("RASPBERRY PI ZERO 2W TARGET TRACKER")
    print("=" * 60)
    print("Performance-optimized for Pi Zero 2W...")
    
    # Override config for Pi Zero optimization
    Config.SHOW_FPS = False  # Reduce display overhead
    Config.SHOW_MASK = False  # Disable mask by default to save CPU
    
    # Step 1: Initialize camera
    try:
        picam2 = initialize_camera_pi_zero()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        print("Make sure the camera is properly connected and enabled.")
        print("Run 'sudo raspi-config' to enable camera if needed.")
        return
    
    # Step 2: Setup minimal frame queue (size=1 for lowest latency)
    frame_queue = Queue(maxsize=1)
    capture_thread = threading.Thread(
        target=camera_capture_thread_optimized, 
        args=(picam2, frame_queue), 
        daemon=True
    )
    capture_thread.start()
    
    # Step 3: Initialize tracking components
    result_logger = ResultLogger("PiCamera_Zero") if Config.ENABLE_RESULT_LOGGING else None
    tracker = TargetTracker(result_logger)
    
    # Step 4: Create display window
    cv2.namedWindow("Pi Zero Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pi Zero Tracker", 320, 240)  # Match camera resolution
    
    # Step 5: Print status and controls
    print("\nPi Zero camera feed started!")
    print(f"Resolution: 320x240 (optimized for Pi Zero)")
    print(f"Target: Red objects, min area: {Config.MIN_AREA} pixels")
    print(f"Result logging: {'ON' if Config.ENABLE_RESULT_LOGGING else 'OFF'}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("  'm' - Toggle mask (impacts performance)")
    print("  'f' - Toggle FPS display")
    print("\nOptimized for real-time performance on Pi Zero 2W...")
    
    frame_count = 0
    last_fps_time = time.time()
    
    try:
        # Step 6: Main processing loop (optimized)
        while True:
            # Get frame from capture queue with minimal delay
            if frame_queue.empty():
                time.sleep(0.001)  # Minimal delay
                continue
                
            frame_rgb = frame_queue.get()
            
            # Convert RGB to BGR (since camera resolution matches tracker, no resize needed)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Process frame through tracker pipeline
            annotated_frame = tracker.process_frame(frame_bgr)
            
            # Display main tracking view
            cv2.imshow("Pi Zero Tracker", annotated_frame)
            
            # Display debug mask only if explicitly enabled
            if Config.SHOW_MASK:
                mask = tracker.get_debug_mask()
                if mask is not None:
                    cv2.imshow("Mask", mask)
            
            # Simple FPS calculation (if enabled)
            frame_count += 1
            if Config.SHOW_FPS and frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - last_fps_time)
                print(f"FPS: {fps:.1f}")
                last_fps_time = current_time
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                # Save snapshot
                filename = f"pi_zero_snapshot_{tracker.frame_count:06d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved: {filename}")
            elif key == ord('m'):
                # Toggle mask view (warn about performance impact)
                Config.SHOW_MASK = not Config.SHOW_MASK
                if Config.SHOW_MASK:
                    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
                    print("Mask view ON (may reduce FPS)")
                else:
                    cv2.destroyWindow("Mask")
                    print("Mask view OFF")
            elif key == ord('f'):
                # Toggle FPS display
                Config.SHOW_FPS = not Config.SHOW_FPS
                print(f"FPS display: {'ON' if Config.SHOW_FPS else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Step 7: Cleanup
        print("Shutting down Pi Zero tracker...")
        
        # Finalize logging session
        if result_logger:
            result_logger.finalize_session(tracker.current_fps)
        
        # Stop camera
        picam2.stop()
        print("Camera stopped")
        
        # Close all windows
        cv2.destroyAllWindows()
        
        print("Pi Zero tracker stopped. Goodbye!")

if __name__ == "__main__":
    main() 