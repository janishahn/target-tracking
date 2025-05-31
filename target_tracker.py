"""
Ultra-Lightweight Target Detection & Tracking Pipeline
Designed for MacBook development (Python + OpenCV), fully portable to Raspberry Pi Zero W

Single-file monolith containing all functionality for real-time target tracking.
"""

import cv2
import numpy as np
import os
import time
import argparse
from typing import Tuple, Optional, List

# =============================================================================
# CONFIGURATION & PARAMETERS
# =============================================================================

class Config:
    """Configuration parameters for target tracking"""
    
    # Video source configuration
    # Switch between video file and camera based on environment variable or argument
    VIDEO_SOURCE = "demo.mp4"  # Default to demo video
    if os.getenv("USE_PI_CAMERA") == "1":
        VIDEO_SOURCE = 0  # Pi's camera index
    
    # Frame dimensions (scaled down for Pi performance)
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    
    # HSV color thresholds for red target detection
    # Red color wraps around in HSV, so we need two ranges
    HSV_LOWER1 = (0, 100, 100)    # Lower red range
    HSV_UPPER1 = (10, 255, 255)
    HSV_LOWER2 = (160, 100, 100)  # Upper red range  
    HSV_UPPER2 = (180, 255, 255)
    
    # Detection filtering parameters
    MIN_AREA = 200              # Minimum contour area in pixels
    MIN_CIRCULARITY = 0.6       # Minimum circularity (0-1)
    MAX_DETECTION_RADIUS = 100  # Maximum detection radius in pixels
    
    # Tracking parameters
    SMOOTHING_ALPHA = 0.3       # EMA smoothing factor (0-1)
    MAX_MISSING_FRAMES = 5      # Frames to wait before declaring target lost
    SEARCH_WINDOW_SIZE = 20     # Pixel radius for tracking window
    
    # Debug and visualization
    DEBUG = True
    SHOW_MASK = False
    SHOW_FPS = True
    
    # Output configuration
    OUTPUT_CSV = False
    UDP_OUTPUT = False
    UDP_PORT = 12345

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def circularity_of(contour) -> float:
    """
    Calculate circularity of a contour.
    Circularity = (4π × area) / (perimeter²)
    Returns value between 0 and 1, where 1 is a perfect circle.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:  # Avoid division by zero
        return 0.0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)  # Cap at 1.0

# =============================================================================
# TARGET TRACKER CLASS (Enhanced Implementation)
# =============================================================================

class TargetTracker:
    """
    Main target tracking class with detection, tracking, and visualization.
    Enhanced implementation following the detailed specification.
    """
    
    def __init__(self):
        # Tracking state variables
        self.last_centroid = None
        self.smoothed_centroid = None
        self.last_radius = 0
        self.smoothed_radius = 0
        self.missing_frames = 0
        self.frame_count = 0
        self.locked = False
        self.lock_duration = 0
        
        # Configuration from Config class
        self.frame_width = Config.FRAME_WIDTH
        self.frame_height = Config.FRAME_HEIGHT
        self.alpha = Config.SMOOTHING_ALPHA
        
        # Precompute morphological kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # HSV ranges for red color detection (handles wraparound)
        self.hsv_ranges = [
            (Config.HSV_LOWER1, Config.HSV_UPPER1),
            (Config.HSV_LOWER2, Config.HSV_UPPER2)
        ]
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        self.last_time = time.time()
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess raw BGR frame from camera or video.
        
        Args:
            frame: Raw BGR frame
            
        Returns:
            Preprocessed frame (resized and optionally blurred)
        """
        # Resize frame to configured dimensions
        frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Optional Gaussian blur for noise reduction (only in debug mode)
        if Config.DEBUG:
            frame_blur = cv2.GaussianBlur(frame_resized, (5, 5), 0)
            return frame_blur
        else:
            return frame_resized
    
    def compute_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """
        Create binary mask for target color detection using HSV thresholding.
        Handles red color which wraps around in HSV space.
        
        Args:
            hsv_frame: HSV converted frame
            
        Returns:
            Binary mask with target pixels as white (255)
        """
        # Create masks for both red ranges (handles HSV wraparound)
        mask1 = cv2.inRange(hsv_frame, self.hsv_ranges[0][0], self.hsv_ranges[0][1])
        mask2 = cv2.inRange(hsv_frame, self.hsv_ranges[1][0], self.hsv_ranges[1][1])
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up noise
        # Open: removes small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Close: fills small holes (optional)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return mask
    
    def find_candidate(self, mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], int, float]]:
        """
        Find the best target candidate from binary mask.
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            
        Returns:
            Best candidate as ((x, y), radius, area) or None if no valid candidate
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        candidates = []
        
        # Evaluate each contour
        for cnt in contours:
            # Check minimum area
            area = cv2.contourArea(cnt)
            if area < Config.MIN_AREA:
                continue
            
            # Check circularity
            circularity = circularity_of(cnt)
            if circularity < Config.MIN_CIRCULARITY:
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            
            # Check maximum radius
            if radius > Config.MAX_DETECTION_RADIUS:
                continue
            
            # Add to candidates list
            candidates.append((cnt, area, circularity, (int(x), int(y), radius)))
        
        if not candidates:
            return None
        
        # Sort by area (descending) and pick the largest
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = candidates[0]
        
        # Return (centroid, radius, area)
        _, area, _, (x, y, radius) = best_candidate
        return ((x, y), radius, area)
    
    def update_track(self, candidate: Optional[Tuple[Tuple[int, int], int, float]]):
        """
        Update tracking state based on detection candidate.
        
        Args:
            candidate: Best candidate from find_candidate or None
        """
        if candidate is not None:
            (x, y), radius, area = candidate
            
            # Check if this is first detection or within search window
            if (self.last_centroid is None or 
                self._is_within_search_window((x, y))):
                
                # Update tracking state
                if self.last_centroid is None:
                    # First detection ever
                    self.smoothed_centroid = (float(x), float(y))
                    self.smoothed_radius = float(radius)
                else:
                    # Apply exponential moving average (EMA)
                    new_cx = (1 - self.alpha) * self.smoothed_centroid[0] + self.alpha * x
                    new_cy = (1 - self.alpha) * self.smoothed_centroid[1] + self.alpha * y
                    new_r = (1 - self.alpha) * self.smoothed_radius + self.alpha * radius
                    
                    self.smoothed_centroid = (new_cx, new_cy)
                    self.smoothed_radius = new_r
                
                # Update state
                self.last_centroid = self.smoothed_centroid
                self.last_radius = self.smoothed_radius
                self.missing_frames = 0
                self.locked = True
                
                if self.lock_duration == 0:
                    self.lock_duration = 1
                else:
                    self.lock_duration += 1
                    
            else:
                # Detection is far from previous centroid
                # For simplicity: treat as reacquisition if area is significantly larger
                if (self.last_radius > 0 and 
                    area > 2 * (np.pi * self.last_radius ** 2)):
                    # Large area suggests this might be the same target
                    # Accept as reacquisition
                    self.smoothed_centroid = (float(x), float(y))
                    self.smoothed_radius = float(radius)
                    self.last_centroid = self.smoothed_centroid
                    self.last_radius = self.smoothed_radius
                    self.missing_frames = 0
                    self.locked = True
                    self.lock_duration = 1
                else:
                    # Ignore this detection, increment missing frames
                    self.missing_frames += 1
                    if self.missing_frames > Config.MAX_MISSING_FRAMES:
                        self.locked = False
                        self.lock_duration = 0
                        self.last_centroid = None
                        self.smoothed_centroid = None
        else:
            # No candidate this frame
            self.missing_frames += 1
            if self.missing_frames > Config.MAX_MISSING_FRAMES:
                self.locked = False
                self.lock_duration = 0
                self.last_centroid = None
                self.smoothed_centroid = None
    
    def _is_within_search_window(self, new_centroid: Tuple[int, int]) -> bool:
        """Check if new detection is within search window of last known position"""
        if self.last_centroid is None:
            return True
        
        dx = new_centroid[0] - self.last_centroid[0]
        dy = new_centroid[1] - self.last_centroid[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance <= Config.SEARCH_WINDOW_SIZE
    
    def get_state(self) -> Tuple[int, int, int, bool]:
        """
        Get current tracking state.
        
        Returns:
            (cx, cy, radius, locked_flag) - (-1, -1, 0, False) if lost
        """
        if self.locked and self.smoothed_centroid is not None:
            cx = int(self.smoothed_centroid[0])
            cy = int(self.smoothed_centroid[1])
            radius = int(self.smoothed_radius)
            return (cx, cy, radius, True)
        else:
            return (-1, -1, 0, False)
    
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """
        Add visualization overlay to frame.
        
        Args:
            frame: BGR frame to annotate
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        cx, cy, radius, locked = self.get_state()
        
        if locked:
            # Draw target circle
            cv2.circle(vis_frame, (cx, cy), radius, (0, 255, 0), 2)
            
            # Draw crosshairs
            cv2.line(vis_frame, (cx-10, cy), (cx+10, cy), (0, 255, 0), 2)
            cv2.line(vis_frame, (cx, cy-10), (cx, cy+10), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, (cx, cy), 2, (0, 255, 0), -1)
            
            # Add text info
            info_text = f"LOCKED ({self.lock_duration} frames)"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show coordinates
            coord_text = f"({cx}, {cy}) r={radius}"
            cv2.putText(vis_frame, coord_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Target lost
            status_text = "TARGET LOST"
            cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show FPS
        if Config.SHOW_FPS:
            current_time = time.time()
            if current_time - self.last_time > 0:
                fps = 1.0 / (current_time - self.last_time)
                self.current_fps = fps
            self.last_time = current_time
            
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(vis_frame, fps_text, (10, vis_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(vis_frame, frame_text, (10, vis_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect, track, and visualize.
        This is the main processing pipeline following the implementation plan.
        
        Args:
            frame: Raw BGR frame
            
        Returns:
            Annotated frame with visualization overlay
        """
        self.frame_count += 1
        
        # Step 1: Preprocess frame
        frame_proc = self.preprocess(frame)
        
        # Step 2: Convert to HSV
        hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
        
        # Step 3: Compute mask
        mask = self.compute_mask(hsv)
        
        # Step 4: Find best candidate
        candidate = self.find_candidate(mask)
        
        # Step 5: Update tracking
        self.update_track(candidate)
        
        # Step 6: Get state for output
        cx, cy, radius, locked = self.get_state()
        
        # Step 7: Output state (CSV format)
        if Config.OUTPUT_CSV:
            print(f"{self.frame_count},{cx},{cy},{radius},{int(locked)}")
        
        # Step 8: Annotate frame for visualization
        vis_frame = self.annotate(frame_proc)
        
        # Store mask for debug visualization
        self._last_mask = mask
        
        return vis_frame
    
    def get_debug_mask(self) -> Optional[np.ndarray]:
        """Get the last computed mask for debug visualization"""
        return getattr(self, '_last_mask', None)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point following the implementation plan"""
    parser = argparse.ArgumentParser(description='Ultra-Lightweight Target Tracker')
    parser.add_argument('--source', type=str, help='Video source (file path or camera index)')
    parser.add_argument('--camera', action='store_true', help='Use camera instead of video file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--csv', action='store_true', help='Output CSV tracking data')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless)')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.source:
        Config.VIDEO_SOURCE = args.source
    elif args.camera:
        Config.VIDEO_SOURCE = 0
    
    if args.debug:
        Config.DEBUG = True
        Config.SHOW_MASK = True
    
    if args.csv:
        Config.OUTPUT_CSV = True
        print("frame,cx,cy,radius,locked")  # CSV header
    
    # Step 1: Initialize VideoCapture
    try:
        if isinstance(Config.VIDEO_SOURCE, str):
            cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
            print(f"Opening video file: {Config.VIDEO_SOURCE}")
        else:
            cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
            print(f"Opening camera: {Config.VIDEO_SOURCE}")
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Debug: Print video properties
    if isinstance(Config.VIDEO_SOURCE, str):
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height}")
    
    # Set camera properties for better performance (only for cameras, not video files)
    if not isinstance(Config.VIDEO_SOURCE, str):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Step 2: Instantiate detector
    detector = TargetTracker()
    
    # Step 3: Create window if DEBUG
    if Config.DEBUG and not args.no_display:
        cv2.namedWindow("Target Tracker", cv2.WINDOW_NORMAL)
        if Config.SHOW_MASK:
            cv2.namedWindow("Color Mask", cv2.WINDOW_NORMAL)
    
    print("Target Tracker started. Press 'q' to quit, 's' to save frame, 'm' to toggle mask view")
    print(f"Target: Red objects with area >= {Config.MIN_AREA} pixels")
    print(f"Frame size: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    
    # Step 4: Main processing loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or failed to read frame (frame {detector.frame_count})")
                break
            
            # Process frame through the complete pipeline
            vis_frame = detector.process_frame(frame)
            
            # Display results (if not headless)
            if not args.no_display:
                if Config.DEBUG:
                    cv2.imshow('Target Tracker', vis_frame)
                    
                    # Show mask in debug mode
                    if Config.SHOW_MASK:
                        mask = detector.get_debug_mask()
                        if mask is not None:
                            cv2.imshow('Color Mask', mask)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"frame_{detector.frame_count:06d}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"Saved frame: {filename}")
                elif key == ord('m'):
                    # Toggle mask view
                    Config.SHOW_MASK = not Config.SHOW_MASK
                    if not Config.SHOW_MASK:
                        cv2.destroyWindow('Color Mask')
                elif key == ord('d'):
                    # Toggle debug mode
                    Config.DEBUG = not Config.DEBUG
                    print(f"Debug mode: {'ON' if Config.DEBUG else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Step 5: Cleanup
        cap.release()
        if Config.DEBUG and not args.no_display:
            cv2.destroyAllWindows()
        print("Target Tracker stopped")

if __name__ == "__main__":
    main() 