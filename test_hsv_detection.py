#!/usr/bin/env python3
"""
HSV Color Detection Test Script
Quick diagnostic tool to test red object detection parameters
"""

import cv2
import numpy as np
import os
from target_tracker import Config

def test_hsv_ranges_on_image(image_path):
    """Test HSV color detection on a sample image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Resize to match tracker resolution
    img_resized = cv2.resize(img, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Test original HSV ranges
    print("Testing ORIGINAL HSV ranges:")
    print(f"Range 1: {Config.HSV_LOWER1} - {Config.HSV_UPPER1}")
    print(f"Range 2: {Config.HSV_LOWER2} - {Config.HSV_UPPER2}")
    
    mask1_orig = cv2.inRange(hsv, Config.HSV_LOWER1, Config.HSV_UPPER1)
    mask2_orig = cv2.inRange(hsv, Config.HSV_LOWER2, Config.HSV_UPPER2)
    mask_orig = cv2.bitwise_or(mask1_orig, mask2_orig)
    
    orig_pixels = cv2.countNonZero(mask_orig)
    print(f"Original ranges detected pixels: {orig_pixels}")
    
    # Test relaxed HSV ranges
    relaxed_lower1 = (0, 50, 50)
    relaxed_upper1 = (15, 255, 255)
    relaxed_lower2 = (155, 50, 50)
    relaxed_upper2 = (180, 255, 255)
    
    print("\nTesting RELAXED HSV ranges:")
    print(f"Range 1: {relaxed_lower1} - {relaxed_upper1}")
    print(f"Range 2: {relaxed_lower2} - {relaxed_upper2}")
    
    mask1_relaxed = cv2.inRange(hsv, relaxed_lower1, relaxed_upper1)
    mask2_relaxed = cv2.inRange(hsv, relaxed_lower2, relaxed_upper2)
    mask_relaxed = cv2.bitwise_or(mask1_relaxed, mask2_relaxed)
    
    relaxed_pixels = cv2.countNonZero(mask_relaxed)
    print(f"Relaxed ranges detected pixels: {relaxed_pixels}")
    
    # Test very permissive ranges
    permissive_lower1 = (0, 30, 30)
    permissive_upper1 = (20, 255, 255)
    permissive_lower2 = (150, 30, 30)
    permissive_upper2 = (180, 255, 255)
    
    print("\nTesting VERY PERMISSIVE HSV ranges:")
    print(f"Range 1: {permissive_lower1} - {permissive_upper1}")
    print(f"Range 2: {permissive_lower2} - {permissive_upper2}")
    
    mask1_perm = cv2.inRange(hsv, permissive_lower1, permissive_upper1)
    mask2_perm = cv2.inRange(hsv, permissive_lower2, permissive_upper2)
    mask_perm = cv2.bitwise_or(mask1_perm, mask2_perm)
    
    perm_pixels = cv2.countNonZero(mask_perm)
    print(f"Permissive ranges detected pixels: {perm_pixels}")
    
    # Save debug images
    output_dir = "hsv_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/original_image.jpg", img_resized)
    cv2.imwrite(f"{output_dir}/mask_original.jpg", mask_orig)
    cv2.imwrite(f"{output_dir}/mask_relaxed.jpg", mask_relaxed)
    cv2.imwrite(f"{output_dir}/mask_permissive.jpg", mask_perm)
    
    print(f"\nDebug images saved to: {output_dir}/")
    print("Check the mask images to see which HSV ranges work best")

def test_with_camera():
    """Test HSV detection with live camera feed"""
    print("Testing with live camera feed...")
    print("Press 's' to save test image, 'q' to quit")
    
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            buffer_count=2
        )
        picam2.configure(camera_config)
        picam2.start()
        
        frame_count = 0
        while True:
            frame_rgb = picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Test HSV detection
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # Use relaxed ranges
            mask1 = cv2.inRange(hsv, (0, 50, 50), (15, 255, 255))
            mask2 = cv2.inRange(hsv, (155, 50, 50), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
            
            detected_pixels = cv2.countNonZero(mask)
            
            # Annotate frame
            cv2.putText(frame_bgr, f"Red pixels: {detected_pixels}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show result (if display available)
            try:
                cv2.imshow("HSV Test", frame_bgr)
                cv2.imshow("Red Mask", mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"test_frame_{frame_count:03d}.jpg", frame_bgr)
                    cv2.imwrite(f"test_mask_{frame_count:03d}.jpg", mask)
                    print(f"Saved test_frame_{frame_count:03d}.jpg and test_mask_{frame_count:03d}.jpg")
                    frame_count += 1
            except:
                # Headless mode - just print stats
                print(f"Frame {frame_count}: Red pixels detected: {detected_pixels}")
                if frame_count % 30 == 0:  # Save every 30 frames
                    cv2.imwrite(f"test_frame_{frame_count:03d}.jpg", frame_bgr)
                    cv2.imwrite(f"test_mask_{frame_count:03d}.jpg", mask)
                    print(f"Saved test_frame_{frame_count:03d}.jpg")
                frame_count += 1
                
                if frame_count > 100:  # Stop after 100 frames in headless mode
                    break
        
        picam2.stop()
        cv2.destroyAllWindows()
        
    except ImportError:
        print("Picamera2 not available. Please run this on a Raspberry Pi with camera.")
    except Exception as e:
        print(f"Camera test failed: {e}")

if __name__ == "__main__":
    import sys
    
    print("HSV Color Detection Test")
    print("=" * 30)
    
    if len(sys.argv) > 1:
        # Test with provided image
        test_hsv_ranges_on_image(sys.argv[1])
    else:
        # Test with camera
        test_with_camera() 