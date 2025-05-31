"""
Test script to verify the color format fix for red vs blue tracking issue.
This script helps diagnose camera color format issues and test the fix.
"""

import cv2
import numpy as np
from target_tracker import TargetTracker, Config

def test_color_detection():
    """Test color detection with both RGB and BGR formats"""
    print("=" * 60)
    print("COLOR FORMAT DETECTION TEST")
    print("=" * 60)
    print("This test helps verify the camera color format fix.")
    print("Make sure you have a RED object visible in your camera view.")
    print("Press 'q' to quit, 'c' to toggle color format, 's' to save test image")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize tracker with camera source
    tracker = TargetTracker(video_source=0)
    
    # Create windows
    cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red Detection Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tracking Result", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame through tracker
            result_frame = tracker.process_frame(frame)
            
            # Get debug mask
            mask = tracker.get_debug_mask()
            
            # Get tracking state
            cx, cy, radius, locked, bbox = tracker.get_state()
            
            # Create info overlay
            info_frame = frame.copy()
            
            # Add color format info
            format_text = f"Color Format: {'RGB->BGR' if tracker.needs_rgb_to_bgr_conversion else 'BGR (no conversion)'}"
            cv2.putText(info_frame, format_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add detection info
            if locked:
                status_text = f"TRACKING RED OBJECT at ({cx}, {cy})"
                color = (0, 255, 0)  # Green
            else:
                status_text = "NO RED OBJECT DETECTED"
                color = (0, 0, 255)  # Red
            
            cv2.putText(info_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add mask pixel count
            if mask is not None:
                mask_pixels = cv2.countNonZero(mask)
                mask_text = f"Red pixels detected: {mask_pixels}"
                cv2.putText(info_frame, mask_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(info_frame, "Press 'c' to toggle color format", (10, info_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(info_frame, "Press 'q' to quit, 's' to save", (10, info_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(info_frame, "Press 'r' to reset detection", (10, info_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display frames
            cv2.imshow("Original Frame", info_frame)
            cv2.imshow("Tracking Result", result_frame)
            
            if mask is not None:
                # Convert mask to 3-channel for better visibility
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Red Detection Mask", mask_colored)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Toggle color format
                tracker.needs_rgb_to_bgr_conversion = not tracker.needs_rgb_to_bgr_conversion
                tracker.color_format_detected = True
                format_str = "RGB->BGR" if tracker.needs_rgb_to_bgr_conversion else "BGR (no conversion)"
                print(f"🔴 Color format toggled to: {format_str}")
            elif key == ord('s'):
                # Save test images
                cv2.imwrite(f"test_original_{frame_count}.jpg", info_frame)
                cv2.imwrite(f"test_result_{frame_count}.jpg", result_frame)
                if mask is not None:
                    cv2.imwrite(f"test_mask_{frame_count}.jpg", mask)
                print(f"Test images saved (frame {frame_count})")
            elif key == ord('r'):
                # Reset tracker
                tracker = TargetTracker(video_source=0)
                print("Tracker reset")
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Color format test completed")

def create_test_pattern():
    """Create a test pattern image with red and blue objects"""
    print("Creating test pattern image...")
    
    # Create test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background
    
    # Add red circle (BGR format)
    cv2.circle(img, (200, 200), 50, (0, 0, 255), -1)  # Red
    cv2.putText(img, "RED", (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add blue circle (BGR format)
    cv2.circle(img, (440, 200), 50, (255, 0, 0), -1)  # Blue
    cv2.putText(img, "BLUE", (410, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(img, "Test Pattern: RED should be tracked, BLUE should be ignored", 
               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "If BLUE is tracked instead, there's a color format issue", 
               (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save test pattern
    cv2.imwrite("test_pattern.jpg", img)
    print("Test pattern saved as 'test_pattern.jpg'")
    
    # Display test pattern
    cv2.imshow("Test Pattern", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Color Format Fix Test Utility")
    print("1. Test with live camera")
    print("2. Create test pattern image")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_color_detection()
    elif choice == "2":
        create_test_pattern()
    else:
        print("Invalid choice") 