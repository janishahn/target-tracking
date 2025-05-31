#!/usr/bin/env python3
"""
Raspberry Pi Camera Integration for Target Tracking
Integrates live Pi camera feed with the existing TargetTracker pipeline
"""

import time
import threading
from queue import Queue

import cv2
from picamera2 import Picamera2
from target_tracker import TargetTracker, ResultLogger, Config

def initialize_camera() -> Picamera2:
    """
    Initialize and configure Picamera2 for video streaming
    Returns configured camera instance ready for capture
    """
    print("Initializing Raspberry Pi camera...")
    picam2 = Picamera2()
    
    # Configure camera for video streaming
    # Using 640x480 for better quality, will be resized internally by tracker
    camera_config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    
    # Start the camera
    picam2.start()
    print("Camera initialized successfully")
    return picam2

def camera_capture_thread(picam2: Picamera2, frame_queue: Queue):
    """
    Background thread for continuous frame capture
    Pushes frames to queue, dropping old frames if queue is full
    """
    print("Starting camera capture thread...")
    while True:
        try:
            # Capture frame as RGB888 array
            frame_rgb = picam2.capture_array("main")
            
            # Try to put frame in queue, drop if full (non-blocking)
            try:
                frame_queue.put_nowait(frame_rgb)
            except:
                # Queue is full, drop this frame to maintain real-time performance
                continue
                
        except Exception as e:
            print(f"Error in capture thread: {e}")
            break

def main():
    """
    Main application loop for Pi camera tracking
    """
    print("=" * 60)
    print("RASPBERRY PI CAMERA TARGET TRACKER")
    print("=" * 60)
    print("Initializing live video feed with red blob tracking...")
    
    # Step 1: Initialize camera
    try:
        picam2 = initialize_camera()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        print("Make sure the camera is properly connected and enabled.")
        return
    
    # Step 2: Setup threaded frame capture
    frame_queue = Queue(maxsize=2)  # Small queue to maintain real-time performance
    capture_thread = threading.Thread(
        target=camera_capture_thread, 
        args=(picam2, frame_queue), 
        daemon=True
    )
    capture_thread.start()
    
    # Step 3: Initialize tracking components
    result_logger = ResultLogger("PiCamera") if Config.ENABLE_RESULT_LOGGING else None
    tracker = TargetTracker(result_logger)
    
    # Step 4: Create display window
    cv2.namedWindow("Live Tracker", cv2.WINDOW_NORMAL)
    
    # Step 5: Print status and controls
    print("\nCamera feed started successfully!")
    print(f"Target: Red objects (HSV ranges: {Config.HSV_LOWER1}-{Config.HSV_UPPER1}, {Config.HSV_LOWER2}-{Config.HSV_UPPER2})")
    print(f"Min area: {Config.MIN_AREA} pixels")
    print(f"Frame processing: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"Result logging: {'ON' if Config.ENABLE_RESULT_LOGGING else 'OFF'}")
    print("\nControls:")
    print("  'q' - Quit application")
    print("  's' - Save current frame snapshot")
    print("  'm' - Toggle mask view (debug)")
    print("  'b' - Toggle bounding box display")
    print("  'd' - Toggle debug mode")
    print("\nPress 'q' to quit...")
    
    # Debug mask window (if enabled)
    if Config.DEBUG and Config.SHOW_MASK:
        cv2.namedWindow("Live Tracker - Mask", cv2.WINDOW_NORMAL)
    
    try:
        # Step 6: Main processing loop
        while True:
            # Get frame from capture queue
            if frame_queue.empty():
                time.sleep(0.005)  # Small delay to prevent busy waiting
                continue
                
            frame_rgb = frame_queue.get()
            
            # Convert RGB to BGR for OpenCV compatibility
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Process frame through tracker pipeline
            annotated_frame = tracker.process_frame(frame_bgr)
            
            # Display main tracking view
            cv2.imshow("Live Tracker", annotated_frame)
            
            # Display debug mask if enabled
            if Config.DEBUG and Config.SHOW_MASK:
                mask = tracker.get_debug_mask()
                if mask is not None:
                    cv2.imshow("Live Tracker - Mask", mask)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord('s'):
                # Save snapshot
                filename = f"snapshot_{tracker.frame_count:06d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot: {filename}")
            elif key == ord('m'):
                # Toggle mask view
                Config.SHOW_MASK = not Config.SHOW_MASK
                if Config.SHOW_MASK and Config.DEBUG:
                    cv2.namedWindow("Live Tracker - Mask", cv2.WINDOW_NORMAL)
                elif not Config.SHOW_MASK:
                    cv2.destroyWindow("Live Tracker - Mask")
                print(f"Mask view: {'ON' if Config.SHOW_MASK else 'OFF'}")
            elif key == ord('b'):
                # Toggle bounding box display
                Config.SHOW_BOUNDING_BOX = not Config.SHOW_BOUNDING_BOX
                print(f"Bounding box display: {'ON' if Config.SHOW_BOUNDING_BOX else 'OFF'}")
            elif key == ord('d'):
                # Toggle debug mode
                Config.DEBUG = not Config.DEBUG
                print(f"Debug mode: {'ON' if Config.DEBUG else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Step 7: Cleanup
        print("Shutting down...")
        
        # Finalize logging session
        if result_logger:
            result_logger.finalize_session(tracker.current_fps)
        
        # Stop camera
        picam2.stop()
        print("Camera stopped")
        
        # Close all windows
        cv2.destroyAllWindows()
        
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main() 