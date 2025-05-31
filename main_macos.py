#!/usr/bin/env python3
"""
macOS Webcam Integration for Target Tracking
Optimized for macOS development using built-in webcam with the existing TargetTracker pipeline
"""

import time
import threading
from queue import Queue

import cv2
from target_tracker import TargetTracker, ResultLogger, Config

def initialize_webcam() -> cv2.VideoCapture:
    """
    Initialize and configure macOS built-in webcam
    Returns configured VideoCapture instance ready for capture
    """
    print("Initializing macOS built-in webcam...")
    
    # Try different camera indices (0 is usually the built-in camera)
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera found at index {camera_index}")
            
            # Configure camera properties for optimal performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # High resolution for macOS
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 720p should work well
            cap.set(cv2.CAP_PROP_FPS, 30)             # Target 30 FPS
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Minimal buffer for low latency
            
            # Verify settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized successfully:")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  Target FPS: {actual_fps}")
            
            return cap
    
    # If no camera found, raise an exception
    raise RuntimeError("No webcam found. Please check that your camera is connected and not in use by another application.")

def webcam_capture_thread(cap: cv2.VideoCapture, frame_queue: Queue):
    """
    Background thread for continuous webcam frame capture
    Pushes frames to queue, dropping old frames if queue is full
    """
    print("Starting webcam capture thread...")
    while True:
        try:
            # Capture frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from webcam")
                break
            
            # Try to put frame in queue, drop if full (non-blocking)
            try:
                frame_queue.put_nowait(frame)
            except:
                # Queue is full, drop this frame to maintain real-time performance
                continue
                
        except Exception as e:
            print(f"Error in capture thread: {e}")
            break
    
    print("Webcam capture thread stopped")

def main():
    """
    Main application loop for macOS webcam tracking
    """
    print("=" * 60)
    print("MACOS WEBCAM TARGET TRACKER")
    print("=" * 60)
    print("Initializing built-in webcam for red blob tracking...")
    
    # Step 1: Initialize webcam
    try:
        cap = initialize_webcam()
    except Exception as e:
        print(f"Failed to initialize webcam: {e}")
        print("Troubleshooting:")
        print("  • Make sure no other apps are using the camera (Zoom, Skype, etc.)")
        print("  • Check System Preferences > Security & Privacy > Camera")
        print("  • Try running: sudo killall VDCAssistant")
        return
    
    # Step 2: Setup threaded frame capture
    frame_queue = Queue(maxsize=3)  # Slightly larger queue for macOS performance
    capture_thread = threading.Thread(
        target=webcam_capture_thread, 
        args=(cap, frame_queue), 
        daemon=True
    )
    capture_thread.start()
    
    # Step 3: Initialize tracking components
    result_logger = ResultLogger("macOS_Webcam") if Config.ENABLE_RESULT_LOGGING else None
    tracker = TargetTracker(result_logger, 0)  # 0 indicates camera source
    
    # Step 4: Create display windows
    cv2.namedWindow("macOS Webcam Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("macOS Webcam Tracker", 960, 540)  # Nice size for macOS
    
    # Step 5: Print status and controls
    print("\nWebcam feed started successfully!")
    print(f"Target: Red objects (HSV ranges: {Config.HSV_LOWER1}-{Config.HSV_UPPER1}, {Config.HSV_LOWER2}-{Config.HSV_UPPER2})")
    print(f"Min area: {Config.MIN_AREA} pixels")
    print(f"Processing resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"Result logging: {'ON' if Config.ENABLE_RESULT_LOGGING else 'OFF'}")
    print("\nControls:")
    print("  'q' - Quit application")
    print("  's' - Save current frame snapshot")
    print("  'm' - Toggle mask view (debug)")
    print("  'b' - Toggle bounding box display")
    print("  'c' - Toggle color format (RGB/BGR)")
    print("  'd' - Toggle debug mode")
    print("  'f' - Toggle FPS display")
    print("  'r' - Reset tracking")
    print("\nOptimized for macOS development with built-in webcam...")
    
    # Debug mask window (if enabled)
    if Config.DEBUG and Config.SHOW_MASK:
        cv2.namedWindow("macOS Webcam Tracker - Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("macOS Webcam Tracker - Mask", 480, 270)
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    
    try:
        # Step 6: Main processing loop
        while True:
            # Get frame from capture queue
            if frame_queue.empty():
                time.sleep(0.001)  # Small delay to prevent busy waiting
                continue
                
            frame_bgr = frame_queue.get()
            
            # Process frame through tracker pipeline
            annotated_frame = tracker.process_frame(frame_bgr)
            
            # Display main tracking view
            cv2.imshow("macOS Webcam Tracker", annotated_frame)
            
            # Display debug mask if enabled
            if Config.DEBUG and Config.SHOW_MASK:
                mask = tracker.get_debug_mask()
                if mask is not None:
                    cv2.imshow("macOS Webcam Tracker - Mask", mask)
            
            # Performance monitoring
            frame_count += 1
            if Config.SHOW_FPS and frame_count % 60 == 0:  # Every 60 frames for smooth updates
                current_time = time.time()
                fps = 60 / (current_time - last_fps_time)
                print(f"Performance: {fps:.1f} FPS, Frame {frame_count}")
                last_fps_time = current_time
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord('s'):
                # Save snapshot
                filename = f"macos_snapshot_{tracker.frame_count:06d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot: {filename}")
            elif key == ord('m'):
                # Toggle mask view
                Config.SHOW_MASK = not Config.SHOW_MASK
                if Config.SHOW_MASK and Config.DEBUG:
                    cv2.namedWindow("macOS Webcam Tracker - Mask", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("macOS Webcam Tracker - Mask", 480, 270)
                elif not Config.SHOW_MASK:
                    cv2.destroyWindow("macOS Webcam Tracker - Mask")
                print(f"Mask view: {'ON' if Config.SHOW_MASK else 'OFF'}")
            elif key == ord('b'):
                # Toggle bounding box display
                Config.SHOW_BOUNDING_BOX = not Config.SHOW_BOUNDING_BOX
                print(f"Bounding box display: {'ON' if Config.SHOW_BOUNDING_BOX else 'OFF'}")
            elif key == ord('c'):
                # Toggle color format conversion
                tracker.needs_rgb_to_bgr_conversion = not tracker.needs_rgb_to_bgr_conversion
                tracker.color_format_detected = True  # Mark as manually set
                format_str = "RGB->BGR" if tracker.needs_rgb_to_bgr_conversion else "BGR (no conversion)"
                print(f"🔴 Color format manually set to: {format_str}")
                print(f"💡 If this fixes the issue, you can set Config.FORCE_COLOR_FORMAT = {tracker.needs_rgb_to_bgr_conversion} to make it permanent")
            elif key == ord('d'):
                # Toggle debug mode
                Config.DEBUG = not Config.DEBUG
                print(f"Debug mode: {'ON' if Config.DEBUG else 'OFF'}")
                # Close mask window if debug disabled
                if not Config.DEBUG:
                    cv2.destroyWindow("macOS Webcam Tracker - Mask")
            elif key == ord('f'):
                # Toggle FPS display
                Config.SHOW_FPS = not Config.SHOW_FPS
                print(f"FPS display: {'ON' if Config.SHOW_FPS else 'OFF'}")
            elif key == ord('r'):
                # Reset tracking
                print("Resetting tracker state...")
                # Re-initialize tracker to reset state
                tracker = TargetTracker(result_logger, 0)  # 0 indicates camera source
                print("Tracker reset complete")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Step 7: Cleanup
        print("Shutting down macOS webcam tracker...")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Session Statistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Session duration: {total_time:.1f} seconds")
        print(f"  Average FPS: {final_fps:.1f}")
        
        # Finalize logging session
        if result_logger:
            result_logger.finalize_session(final_fps)
        
        # Stop camera
        cap.release()
        print("Webcam released")
        
        # Close all windows
        cv2.destroyAllWindows()
        
        print("macOS webcam tracker stopped. Goodbye!")

if __name__ == "__main__":
    main() 