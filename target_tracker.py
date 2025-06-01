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
import json
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
from threading import Thread
from queue import Queue

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
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # HSV color thresholds for red target detection
    # Red color wraps around in HSV, so we need two ranges
    # BALANCED: Good compromise between accuracy and detection capability
    HSV_LOWER1 = (0, 40, 30)      # Lower red range (balanced thresholds)
    HSV_UPPER1 = (15, 255, 255)   # Wider hue range for better detection
    HSV_LOWER2 = (165, 40, 30)    # Upper red range (safe from purple-blue)  
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
    SHOW_BOUNDING_BOX = True    # Show bounding box instead of just circle
    
    # Color format override (set to force specific behavior)
    # None = auto-detect, True = force RGB->BGR conversion, False = force BGR (no conversion)
    FORCE_COLOR_FORMAT = None
    
    # Output configuration
    OUTPUT_CSV = False
    UDP_OUTPUT = False
    UDP_PORT = 12345
    
    # Result logging configuration
    ENABLE_RESULT_LOGGING = True
    LOG_DIRECTORY = "tracking_results"

# =============================================================================
# RESULT LOGGING CLASS
# =============================================================================

class ResultLogger:
    """
    Comprehensive result logging for target tracking sessions.
    Collects tracking data and generates human-readable reports.
    """
    
    def __init__(self, video_source: str):
        self.video_source = video_source
        self.session_start = datetime.now()
        self.tracking_data = []
        self.session_stats = {
            'total_frames': 0,
            'frames_with_detection': 0,
            'frames_locked': 0,
            'total_lock_duration': 0,
            'max_lock_duration': 0,
            'lock_sessions': [],
            'avg_fps': 0,
            'processing_time': 0
        }
        self.current_lock_session = None
        
        # Create log directory if it doesn't exist
        if Config.ENABLE_RESULT_LOGGING:
            os.makedirs(Config.LOG_DIRECTORY, exist_ok=True)
    
    def log_frame(self, frame_num: int, cx: int, cy: int, radius: int, 
                  locked: bool, bbox: Tuple[int, int, int, int], 
                  lock_duration: int, fps: float):
        """Log data for a single frame"""
        if not Config.ENABLE_RESULT_LOGGING:
            return
            
        frame_data = {
            'frame': frame_num,
            'timestamp': time.time(),
            'centroid': (cx, cy),
            'radius': radius,
            'locked': locked,
            'bbox': bbox,
            'lock_duration': lock_duration,
            'fps': fps
        }
        
        self.tracking_data.append(frame_data)
        
        # Update session statistics
        self.session_stats['total_frames'] = frame_num
        
        if cx != -1 and cy != -1:  # Valid detection
            self.session_stats['frames_with_detection'] += 1
        
        if locked:
            self.session_stats['frames_locked'] += 1
            self.session_stats['max_lock_duration'] = max(
                self.session_stats['max_lock_duration'], lock_duration
            )
            
            # Track lock sessions
            if self.current_lock_session is None:
                self.current_lock_session = {
                    'start_frame': frame_num,
                    'start_position': (cx, cy),
                    'duration': 0,
                    'avg_position': [cx, cy],
                    'position_variance': 0
                }
            else:
                self.current_lock_session['duration'] = lock_duration
                # Update average position
                self.current_lock_session['avg_position'][0] = (
                    self.current_lock_session['avg_position'][0] * 0.9 + cx * 0.1
                )
                self.current_lock_session['avg_position'][1] = (
                    self.current_lock_session['avg_position'][1] * 0.9 + cy * 0.1
                )
        else:
            # End current lock session if it exists
            if self.current_lock_session is not None:
                self.current_lock_session['end_frame'] = frame_num - 1
                self.session_stats['lock_sessions'].append(self.current_lock_session.copy())
                self.session_stats['total_lock_duration'] += self.current_lock_session['duration']
                self.current_lock_session = None
    
    def finalize_session(self, final_fps: float):
        """Finalize the logging session and generate reports"""
        if not Config.ENABLE_RESULT_LOGGING:
            return
            
        # Close any ongoing lock session
        if self.current_lock_session is not None:
            self.current_lock_session['end_frame'] = self.session_stats['total_frames']
            self.session_stats['lock_sessions'].append(self.current_lock_session.copy())
            self.session_stats['total_lock_duration'] += self.current_lock_session['duration']
        
        # Calculate final statistics
        self.session_stats['avg_fps'] = final_fps
        self.session_stats['processing_time'] = (datetime.now() - self.session_start).total_seconds()
        
        # Generate reports
        self._generate_summary_report()
        self._generate_detailed_report()
        self._generate_json_report()
        
        print(f"\nTracking results saved to: {Config.LOG_DIRECTORY}/")
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.LOG_DIRECTORY}/{timestamp}_tracking_summary.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TARGET TRACKING SESSION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Session info
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video Source: {self.video_source}\n")
            f.write(f"Processing Time: {self.session_stats['processing_time']:.2f} seconds\n")
            f.write(f"Average FPS: {self.session_stats['avg_fps']:.1f}\n\n")
            
            # Frame statistics
            f.write("FRAME STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Frames Processed: {self.session_stats['total_frames']}\n")
            f.write(f"Frames with Detection: {self.session_stats['frames_with_detection']}\n")
            f.write(f"Frames with Target Lock: {self.session_stats['frames_locked']}\n")
            
            if self.session_stats['total_frames'] > 0:
                detection_rate = (self.session_stats['frames_with_detection'] / 
                                self.session_stats['total_frames']) * 100
                lock_rate = (self.session_stats['frames_locked'] / 
                           self.session_stats['total_frames']) * 100
                f.write(f"Detection Rate: {detection_rate:.1f}%\n")
                f.write(f"Lock Rate: {lock_rate:.1f}%\n")
            
            f.write("\n")
            
            # Lock session statistics
            f.write("LOCK SESSION STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Number of Lock Sessions: {len(self.session_stats['lock_sessions'])}\n")
            f.write(f"Total Lock Duration: {self.session_stats['total_lock_duration']} frames\n")
            f.write(f"Maximum Lock Duration: {self.session_stats['max_lock_duration']} frames\n")
            
            if len(self.session_stats['lock_sessions']) > 0:
                avg_lock_duration = (self.session_stats['total_lock_duration'] / 
                                   len(self.session_stats['lock_sessions']))
                f.write(f"Average Lock Duration: {avg_lock_duration:.1f} frames\n")
            
            f.write("\n")
            
            # Individual lock sessions
            if len(self.session_stats['lock_sessions']) > 0:
                f.write("INDIVIDUAL LOCK SESSIONS:\n")
                f.write("-" * 26 + "\n")
                for i, session in enumerate(self.session_stats['lock_sessions'], 1):
                    f.write(f"Session {i}:\n")
                    f.write(f"  Frames: {session['start_frame']} - {session.get('end_frame', 'ongoing')}\n")
                    f.write(f"  Duration: {session['duration']} frames\n")
                    f.write(f"  Start Position: ({session['start_position'][0]}, {session['start_position'][1]})\n")
                    f.write(f"  Average Position: ({session['avg_position'][0]:.1f}, {session['avg_position'][1]:.1f})\n")
                    f.write("\n")
            
            # Configuration used
            f.write("CONFIGURATION USED:\n")
            f.write("-" * 19 + "\n")
            f.write(f"Frame Size: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}\n")
            f.write(f"Min Area: {Config.MIN_AREA} pixels\n")
            f.write(f"Min Circularity: {Config.MIN_CIRCULARITY}\n")
            f.write(f"Max Detection Radius: {Config.MAX_DETECTION_RADIUS} pixels\n")
            f.write(f"Smoothing Alpha: {Config.SMOOTHING_ALPHA}\n")
            f.write(f"Max Missing Frames: {Config.MAX_MISSING_FRAMES}\n")
            f.write(f"Search Window Size: {Config.SEARCH_WINDOW_SIZE} pixels\n")
        
        print(f"Summary report saved: {filename}")
    
    def _generate_detailed_report(self):
        """Generate a detailed frame-by-frame report"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.LOG_DIRECTORY}/{timestamp}_tracking_detailed.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED FRAME-BY-FRAME TRACKING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Format: Frame | Status | Center (x,y) | Radius | BBox (x,y,w,h) | Lock Duration | FPS\n")
            f.write("-" * 80 + "\n")
            
            for data in self.tracking_data:
                status = "LOCKED" if data['locked'] else "LOST  "
                cx, cy = data['centroid']
                bbox_x, bbox_y, bbox_w, bbox_h = data['bbox']
                
                f.write(f"{data['frame']:5d} | {status} | "
                       f"({cx:3d},{cy:3d}) | {data['radius']:3d} | "
                       f"({bbox_x:3d},{bbox_y:3d},{bbox_w:3d},{bbox_h:3d}) | "
                       f"{data['lock_duration']:3d} | {data['fps']:5.1f}\n")
        
        print(f"Detailed report saved: {filename}")
    
    def _generate_json_report(self):
        """Generate a machine-readable JSON report"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.LOG_DIRECTORY}/{timestamp}_tracking_data.json"
        
        report_data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'video_source': self.video_source,
                'processing_time_seconds': self.session_stats['processing_time'],
                'average_fps': self.session_stats['avg_fps']
            },
            'statistics': self.session_stats,
            'configuration': {
                'frame_width': Config.FRAME_WIDTH,
                'frame_height': Config.FRAME_HEIGHT,
                'min_area': Config.MIN_AREA,
                'min_circularity': Config.MIN_CIRCULARITY,
                'max_detection_radius': Config.MAX_DETECTION_RADIUS,
                'smoothing_alpha': Config.SMOOTHING_ALPHA,
                'max_missing_frames': Config.MAX_MISSING_FRAMES,
                'search_window_size': Config.SEARCH_WINDOW_SIZE
            },
            'frame_data': self.tracking_data
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"JSON data saved: {filename}")

# =============================================================================
# TARGET TRACKER CLASS (Enhanced Implementation)
# =============================================================================

class TargetTracker:
    """
    Main target tracking class with detection, tracking, and visualization.
    Enhanced implementation following the detailed specification.
    """
    
    def __init__(self, result_logger: Optional[ResultLogger] = None, video_source=None):
        # Tracking state variables
        self.last_centroid = None
        self.smoothed_centroid = None
        self.last_radius = 0
        self.smoothed_radius = 0
        self.missing_frames = 0
        self.frame_count = 0
        self.locked = False
        self.lock_duration = 0
        
        # Bounding box tracking variables
        self.last_bbox = None  # (x, y, w, h)
        self.smoothed_bbox = None
        self.last_contour = None
        
        # Configuration from Config class
        self.frame_width = Config.FRAME_WIDTH
        self.frame_height = Config.FRAME_HEIGHT
        self.alpha = Config.SMOOTHING_ALPHA
        
        # Camera color format detection
        self.video_source = video_source
        self.is_camera_source = isinstance(video_source, int) or video_source == 0
        self.color_format_detected = False
        self.needs_rgb_to_bgr_conversion = False
        
        # Pre-allocated buffers for performance optimization
        self.hsv_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.mask_prealloc = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.bgr_frame_prealloc = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.rgb_conversion_buffer = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Precompute morphological kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Create SimpleBlobDetector for efficient blob detection
        params = cv2.SimpleBlobDetector_Params()
        
        # Configure thresholding for binary mask input
        params.minThreshold = 127
        params.maxThreshold = 255
        params.thresholdStep = 10
        
        # Area filtering
        params.filterByArea = True
        params.minArea = Config.MIN_AREA
        params.maxArea = (Config.MAX_DETECTION_RADIUS ** 2) * np.pi  # approximate circle area
        
        # Circularity filtering
        params.filterByCircularity = True
        params.minCircularity = Config.MIN_CIRCULARITY
        
        # Disable other filters for now
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = False
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
        
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
        
        # Result logging
        self.result_logger = result_logger
        
        # ROI-based search optimization
        self.roi_enabled = False
        self.roi_rect = (0, 0, self.frame_width, self.frame_height)  # x, y, w, h
        self.frames_since_full_search = 0
    
    def _detect_camera_color_format(self, frame: np.ndarray) -> bool:
        """
        Detect if camera provides RGB or BGR frames by analyzing color distribution.
        This method assumes we're looking for red objects and checks which format
        produces better red detection in HSV space.
        
        Args:
            frame: Raw frame from camera
            
        Returns:
            True if RGB->BGR conversion is needed, False if frame is already BGR
        """
        if self.color_format_detected:
            return self.needs_rgb_to_bgr_conversion
        
        # Check for manual override first
        if Config.FORCE_COLOR_FORMAT is not None:
            self.needs_rgb_to_bgr_conversion = Config.FORCE_COLOR_FORMAT
            self.color_format_detected = True
            format_str = "RGB->BGR" if Config.FORCE_COLOR_FORMAT else "BGR (no conversion)"
            print(f"🔴 Camera color format FORCED to: {format_str}")
            return self.needs_rgb_to_bgr_conversion
        
        # Only perform detection for camera sources
        if not self.is_camera_source:
            self.color_format_detected = True
            self.needs_rgb_to_bgr_conversion = False
            return False
        
        # Resize frame for analysis
        test_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Test 1: Assume frame is BGR (no conversion)
        hsv_bgr = cv2.cvtColor(test_frame, cv2.COLOR_BGR2HSV)
        mask_bgr_1 = cv2.inRange(hsv_bgr, self.hsv_ranges[0][0], self.hsv_ranges[0][1])
        mask_bgr_2 = cv2.inRange(hsv_bgr, self.hsv_ranges[1][0], self.hsv_ranges[1][1])
        mask_bgr = cv2.bitwise_or(mask_bgr_1, mask_bgr_2)
        red_pixels_bgr = cv2.countNonZero(mask_bgr)
        
        # Test 2: Assume frame is RGB (convert to BGR first)
        test_frame_converted = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)
        hsv_rgb = cv2.cvtColor(test_frame_converted, cv2.COLOR_BGR2HSV)
        mask_rgb_1 = cv2.inRange(hsv_rgb, self.hsv_ranges[0][0], self.hsv_ranges[0][1])
        mask_rgb_2 = cv2.inRange(hsv_rgb, self.hsv_ranges[1][0], self.hsv_ranges[1][1])
        mask_rgb = cv2.bitwise_or(mask_rgb_1, mask_rgb_2)
        red_pixels_rgb = cv2.countNonZero(mask_rgb)
        
        # Determine which format produces more red pixels
        # Add a threshold to avoid false positives from noise
        min_threshold = 50  # Minimum pixels to consider a valid detection
        
        if red_pixels_rgb > red_pixels_bgr and red_pixels_rgb > min_threshold:
            self.needs_rgb_to_bgr_conversion = True
            print(f"🔴 Camera color format detected: RGB (converting to BGR)")
            print(f"  RGB->BGR conversion: {red_pixels_rgb} red pixels")
            print(f"  Direct BGR: {red_pixels_bgr} red pixels")
            print(f"  ✅ This should fix blue objects being tracked instead of red!")
        elif red_pixels_bgr > min_threshold:
            self.needs_rgb_to_bgr_conversion = False
            print(f"🔴 Camera color format detected: BGR (no conversion needed)")
            print(f"  Direct BGR: {red_pixels_bgr} red pixels")
            print(f"  RGB->BGR conversion: {red_pixels_rgb} red pixels")
        else:
            # Fallback: if no clear winner, assume RGB format (common on macOS)
            self.needs_rgb_to_bgr_conversion = True
            print(f"🔴 Camera color format unclear - defaulting to RGB->BGR conversion")
            print(f"  RGB->BGR conversion: {red_pixels_rgb} red pixels")
            print(f"  Direct BGR: {red_pixels_bgr} red pixels")
            print(f"  ⚠️  Warning: No clear red objects detected. Place a red object in view for better detection.")
            print(f"  💡 If tracking still fails, try pressing 'r' to reset with a red object visible.")
        
        self.color_format_detected = True
        return self.needs_rgb_to_bgr_conversion
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess raw frame from camera or video with automatic color format detection.
        Uses pre-allocated buffer to avoid memory allocation overhead.
        
        Args:
            frame: Raw frame (could be RGB or BGR depending on source)
            
        Returns:
            Preprocessed BGR frame (resized and optionally blurred) - pre-allocated buffer
        """
        # Detect and handle color format for camera sources
        if self.is_camera_source and not self.color_format_detected:
            self._detect_camera_color_format(frame)
        
        # Apply color conversion if needed
        if self.needs_rgb_to_bgr_conversion:
            # Convert RGB to BGR using pre-allocated buffer
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, dst=self.rgb_conversion_buffer)
            source_frame = self.rgb_conversion_buffer
        else:
            source_frame = frame
        
        # Resize frame to configured dimensions using pre-allocated buffer
        cv2.resize(source_frame, (self.frame_width, self.frame_height), dst=self.bgr_frame_prealloc)
        
        # Optional Gaussian blur for noise reduction (only in debug mode)
        if Config.DEBUG:
            cv2.GaussianBlur(self.bgr_frame_prealloc, (5, 5), 0, dst=self.bgr_frame_prealloc)
        
        return self.bgr_frame_prealloc
    
    def compute_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """
        Create binary mask for target color detection using HSV thresholding.
        Handles red color which wraps around in HSV space.
        Uses pre-allocated buffer to avoid memory allocation overhead.
        
        Args:
            hsv_frame: HSV converted frame
            
        Returns:
            Binary mask with target pixels as white (255) - pre-allocated buffer
        """
        # Create first mask for red range using pre-allocated buffer
        cv2.inRange(hsv_frame, self.hsv_ranges[0][0], self.hsv_ranges[0][1], dst=self.mask_prealloc)
        
        # Create second mask for red range (handles HSV wraparound)
        mask2 = cv2.inRange(hsv_frame, self.hsv_ranges[1][0], self.hsv_ranges[1][1])
        
        # Combine masks using pre-allocated buffer
        cv2.bitwise_or(self.mask_prealloc, mask2, dst=self.mask_prealloc)
        
        # Morphological operations to clean up noise (in-place)
        # Open: removes small blobs
        cv2.morphologyEx(self.mask_prealloc, cv2.MORPH_OPEN, self.morph_kernel, dst=self.mask_prealloc)
        
        # Close: fills small holes (optional)
        cv2.morphologyEx(self.mask_prealloc, cv2.MORPH_CLOSE, self.morph_kernel, dst=self.mask_prealloc)
        
        return self.mask_prealloc
    
    def _compute_roi_mask(self, hsv_roi: np.ndarray) -> np.ndarray:
        """
        Create binary mask for ROI region using HSV thresholding.
        This is a separate method to handle ROI-specific masking without buffer conflicts.
        
        Args:
            hsv_roi: HSV converted ROI region
            
        Returns:
            Binary mask for ROI with target pixels as white (255)
        """
        # Create masks for both red ranges (handles HSV wraparound)
        mask1 = cv2.inRange(hsv_roi, self.hsv_ranges[0][0], self.hsv_ranges[0][1])
        mask2 = cv2.inRange(hsv_roi, self.hsv_ranges[1][0], self.hsv_ranges[1][1])
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up noise
        # Open: removes small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Close: fills small holes (optional)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return mask
    
    def find_candidate(self, mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], int, float, Tuple[int, int, int, int], np.ndarray]]:
        """
        Find the best target candidate from binary mask using SimpleBlobDetector.
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            
        Returns:
            Best candidate as ((x, y), radius, area, bbox, contour) or None if no valid candidate
            bbox format: (x, y, w, h) - top-left corner and dimensions
            Note: contour is None since SimpleBlobDetector doesn't provide contours
        """
        # Use SimpleBlobDetector for efficient blob detection
        keypoints = self.blob_detector.detect(mask)
        
        if not keypoints:
            return None
        
        # Select the largest keypoint by size (diameter)
        best_kp = max(keypoints, key=lambda kp: kp.size)
        
        # Extract blob properties
        x, y = best_kp.pt
        radius = best_kp.size / 2.0
        area_est = np.pi * radius * radius
        
        # Create bounding box from circle
        bbox_x = int(x - radius)
        bbox_y = int(y - radius)
        bbox_w = int(2 * radius)
        bbox_h = int(2 * radius)
        
        # Return (centroid, radius, area, bbox, contour)
        # Note: contour is None since SimpleBlobDetector doesn't provide contours
        return ((int(x), int(y)), int(radius), area_est, (bbox_x, bbox_y, bbox_w, bbox_h), None)
    
    def update_track(self, candidate: Optional[Tuple[Tuple[int, int], int, float, Tuple[int, int, int, int], np.ndarray]]):
        """
        Update tracking state based on detection candidate.
        
        Args:
            candidate: Best candidate from find_candidate or None
        """
        if candidate is not None:
            (x, y), radius, area, bbox, contour = candidate
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            
            # Check if this is first detection or within search window
            if (self.last_centroid is None or 
                self._is_within_search_window((x, y))):
                
                # Update tracking state
                if self.last_centroid is None:
                    # First detection ever
                    self.smoothed_centroid = (float(x), float(y))
                    self.smoothed_radius = float(radius)
                    self.smoothed_bbox = (float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h))
                else:
                    # Apply exponential moving average (EMA)
                    new_cx = (1 - self.alpha) * self.smoothed_centroid[0] + self.alpha * x
                    new_cy = (1 - self.alpha) * self.smoothed_centroid[1] + self.alpha * y
                    new_r = (1 - self.alpha) * self.smoothed_radius + self.alpha * radius
                    
                    # Smooth bounding box coordinates
                    new_bx = (1 - self.alpha) * self.smoothed_bbox[0] + self.alpha * bbox_x
                    new_by = (1 - self.alpha) * self.smoothed_bbox[1] + self.alpha * bbox_y
                    new_bw = (1 - self.alpha) * self.smoothed_bbox[2] + self.alpha * bbox_w
                    new_bh = (1 - self.alpha) * self.smoothed_bbox[3] + self.alpha * bbox_h
                    
                    self.smoothed_centroid = (new_cx, new_cy)
                    self.smoothed_radius = new_r
                    self.smoothed_bbox = (new_bx, new_by, new_bw, new_bh)
                
                # Update state
                self.last_centroid = self.smoothed_centroid
                self.last_radius = self.smoothed_radius
                self.last_bbox = self.smoothed_bbox
                self.last_contour = contour
                self.missing_frames = 0
                self.locked = True
                
                if self.lock_duration == 0:
                    self.lock_duration = 1
                else:
                    self.lock_duration += 1
                
                # Update ROI for next frame
                center_x, center_y = int(self.smoothed_centroid[0]), int(self.smoothed_centroid[1])
                half_win = Config.SEARCH_WINDOW_SIZE
                new_x = max(0, center_x - half_win)
                new_y = max(0, center_y - half_win)
                new_w = min(self.frame_width - new_x, 2 * half_win)
                new_h = min(self.frame_height - new_y, 2 * half_win)
                self.roi_rect = (new_x, new_y, new_w, new_h)
                self.roi_enabled = True
                self.frames_since_full_search = 0
                    
            else:
                # Detection is far from previous centroid
                # For simplicity: treat as reacquisition if area is significantly larger
                if (self.last_radius > 0 and 
                    area > 2 * (np.pi * self.last_radius ** 2)):
                    # Large area suggests this might be the same target
                    # Accept as reacquisition
                    self.smoothed_centroid = (float(x), float(y))
                    self.smoothed_radius = float(radius)
                    self.smoothed_bbox = (float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h))
                    self.last_centroid = self.smoothed_centroid
                    self.last_radius = self.smoothed_radius
                    self.last_bbox = self.smoothed_bbox
                    self.last_contour = contour
                    self.missing_frames = 0
                    self.locked = True
                    self.lock_duration = 1
                    
                    # Update ROI for reacquisition
                    center_x, center_y = int(self.smoothed_centroid[0]), int(self.smoothed_centroid[1])
                    half_win = Config.SEARCH_WINDOW_SIZE
                    new_x = max(0, center_x - half_win)
                    new_y = max(0, center_y - half_win)
                    new_w = min(self.frame_width - new_x, 2 * half_win)
                    new_h = min(self.frame_height - new_y, 2 * half_win)
                    self.roi_rect = (new_x, new_y, new_w, new_h)
                    self.roi_enabled = True
                    self.frames_since_full_search = 0
                else:
                    # Ignore this detection, increment missing frames
                    self.missing_frames += 1
                    self.frames_since_full_search += 1
                    if self.missing_frames > Config.MAX_MISSING_FRAMES:
                        self.locked = False
                        self.lock_duration = 0
                        self.last_centroid = None
                        self.smoothed_centroid = None
                        self.last_bbox = None
                        self.smoothed_bbox = None
                        self.last_contour = None
                        self.roi_enabled = False
        else:
            # No candidate this frame
            self.missing_frames += 1
            self.frames_since_full_search += 1
            if self.missing_frames > Config.MAX_MISSING_FRAMES:
                self.locked = False
                self.lock_duration = 0
                self.last_centroid = None
                self.smoothed_centroid = None
                self.last_bbox = None
                self.smoothed_bbox = None
                self.last_contour = None
                self.roi_enabled = False
    
    def _is_within_search_window(self, new_centroid: Tuple[int, int]) -> bool:
        """Check if new detection is within search window of last known position"""
        if self.last_centroid is None:
            return True
        
        dx = new_centroid[0] - self.last_centroid[0]
        dy = new_centroid[1] - self.last_centroid[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance <= Config.SEARCH_WINDOW_SIZE
    
    def get_state(self) -> Tuple[int, int, int, bool, Tuple[int, int, int, int]]:
        """
        Get current tracking state.
        
        Returns:
            (cx, cy, radius, locked_flag, bbox) - (-1, -1, 0, False, (-1, -1, 0, 0)) if lost
            bbox format: (x, y, w, h) - top-left corner and dimensions
        """
        if self.locked and self.smoothed_centroid is not None and self.smoothed_bbox is not None:
            cx = int(self.smoothed_centroid[0])
            cy = int(self.smoothed_centroid[1])
            radius = int(self.smoothed_radius)
            bbox = (int(self.smoothed_bbox[0]), int(self.smoothed_bbox[1]), 
                   int(self.smoothed_bbox[2]), int(self.smoothed_bbox[3]))
            return (cx, cy, radius, True, bbox)
        else:
            return (-1, -1, 0, False, (-1, -1, 0, 0))
    
    def calculate_distance_from_center(self, cx: int, cy: int) -> Tuple[float, int, int, int, int]:
        """
        Calculate distance and offset of object center from frame center.
        
        Args:
            cx, cy: Object center coordinates
            
        Returns:
            (absolute_distance, x_offset, y_offset, frame_center_x, frame_center_y)
            - absolute_distance: Euclidean distance in pixels
            - x_offset: Horizontal offset (positive = east/right, negative = west/left)
            - y_offset: Vertical offset (positive = south/down, negative = north/up)
            - frame_center_x, frame_center_y: Frame center coordinates for reference
        """
        frame_center_x = Config.FRAME_WIDTH // 2
        frame_center_y = Config.FRAME_HEIGHT // 2
        
        x_offset = cx - frame_center_x
        y_offset = cy - frame_center_y
        
        absolute_distance = np.sqrt(x_offset**2 + y_offset**2)
        
        return (absolute_distance, x_offset, y_offset, frame_center_x, frame_center_y)
    
    def get_state_with_distance(self) -> Tuple[int, int, int, bool, Tuple[int, int, int, int], Tuple[float, int, int, int, int]]:
        """
        Get current tracking state including distance information.
        
        Returns:
            (cx, cy, radius, locked_flag, bbox, distance_info)
            - distance_info: (abs_distance, x_offset, y_offset, center_x, center_y)
            - Returns (-1, -1, 0, False, (-1, -1, 0, 0), (0.0, 0, 0, center_x, center_y)) if lost
        """
        frame_center_x = Config.FRAME_WIDTH // 2
        frame_center_y = Config.FRAME_HEIGHT // 2
        
        if self.locked and self.smoothed_centroid is not None and self.smoothed_bbox is not None:
            cx = int(self.smoothed_centroid[0])
            cy = int(self.smoothed_centroid[1])
            radius = int(self.smoothed_radius)
            bbox = (int(self.smoothed_bbox[0]), int(self.smoothed_bbox[1]), 
                   int(self.smoothed_bbox[2]), int(self.smoothed_bbox[3]))
            
            distance_info = self.calculate_distance_from_center(cx, cy)
            
            return (cx, cy, radius, True, bbox, distance_info)
        else:
            return (-1, -1, 0, False, (-1, -1, 0, 0), (0.0, 0, 0, frame_center_x, frame_center_y))

    def get_directional_description(self, x_offset: int, y_offset: int) -> str:
        """
        Generate human-readable directional description of object position relative to frame center.
        
        Args:
            x_offset: Horizontal offset (positive = east/right, negative = west/left)
            y_offset: Vertical offset (positive = south/down, negative = north/up)
            
        Returns:
            Human-readable description like "15 pixels north and 8 pixels east"
        """
        if x_offset == 0 and y_offset == 0:
            return "at frame center"
        
        parts = []
        
        # Vertical component
        if y_offset < 0:
            parts.append(f"{abs(y_offset)} pixels north")
        elif y_offset > 0:
            parts.append(f"{y_offset} pixels south")
        
        # Horizontal component
        if x_offset < 0:
            parts.append(f"{abs(x_offset)} pixels west")
        elif x_offset > 0:
            parts.append(f"{x_offset} pixels east")
        
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        elif len(parts) == 1:
            return parts[0]
        else:
            return "at frame center"
    
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """
        Add visualization overlay to frame (in-place for performance).
        
        Args:
            frame: BGR frame to annotate (will be modified in-place)
            
        Returns:
            Same frame object with annotations added
        """
        frame_for_display = frame  # No copy - annotate in-place
        cx, cy, radius, locked, bbox, distance_info = self.get_state_with_distance()
        abs_distance, x_offset, y_offset, frame_center_x, frame_center_y = distance_info
        
        # Always draw frame center crosshairs for reference
        cv2.line(frame_for_display, (frame_center_x-8, frame_center_y), (frame_center_x+8, frame_center_y), (255, 255, 255), 1)
        cv2.line(frame_for_display, (frame_center_x, frame_center_y-8), (frame_center_x, frame_center_y+8), (255, 255, 255), 1)
        cv2.circle(frame_for_display, (frame_center_x, frame_center_y), 3, (255, 255, 255), 1)
        
        if locked:
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            
            if Config.SHOW_BOUNDING_BOX:
                # Draw bounding box
                cv2.rectangle(frame_for_display, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 2)
                
                # Draw target circle (optional, for reference)
                cv2.circle(frame_for_display, (cx, cy), radius, (0, 255, 0), 1)
            else:
                # Draw target circle only
                cv2.circle(frame_for_display, (cx, cy), radius, (0, 255, 0), 2)
            
            # Draw crosshairs
            cv2.line(frame_for_display, (cx-10, cy), (cx+10, cy), (0, 255, 0), 2)
            cv2.line(frame_for_display, (cx, cy-10), (cx, cy+10), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame_for_display, (cx, cy), 2, (0, 255, 0), -1)
            
            # Draw line from frame center to object center
            cv2.line(frame_for_display, (frame_center_x, frame_center_y), (cx, cy), (255, 255, 0), 1)
            
            # Add bounding box coordinates at top-left corner of the box (only if showing bounding box)
            if Config.SHOW_BOUNDING_BOX:
                bbox_coord_text = f"({bbox_x},{bbox_y})"
                # Position text slightly above the bounding box
                text_x = bbox_x
                text_y = max(bbox_y - 5, 15)  # Ensure text doesn't go off screen
                
                # Add background rectangle for better text visibility
                text_size = cv2.getTextSize(bbox_coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame_for_display, (text_x, text_y - text_size[1] - 2), 
                             (text_x + text_size[0] + 4, text_y + 2), (0, 0, 0), -1)
                
                # Draw the coordinate text
                cv2.putText(frame_for_display, bbox_coord_text, (text_x + 2, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Add bounding box dimensions next to coordinates
                bbox_size_text = f"{bbox_w}x{bbox_h}"
                cv2.putText(frame_for_display, bbox_size_text, (text_x + 2, text_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Add text info (status)
            info_text = f"LOCKED ({self.lock_duration} frames)"
            cv2.putText(frame_for_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show center coordinates
            coord_text = f"Center: ({cx}, {cy})"
            cv2.putText(frame_for_display, coord_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show bounding box info
            bbox_info_text = f"BBox: ({bbox_x},{bbox_y}) {bbox_w}x{bbox_h}"
            cv2.putText(frame_for_display, bbox_info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show distance information
            distance_text = f"Distance: {abs_distance:.1f}px"
            cv2.putText(frame_for_display, distance_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show offset information
            offset_text = f"Offset: ({x_offset:+d}, {y_offset:+d})"
            cv2.putText(frame_for_display, offset_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show directional description (if there's space)
            if frame_for_display.shape[0] > 140:  # Only show if frame is tall enough
                direction_text = self.get_directional_description(x_offset, y_offset)
                cv2.putText(frame_for_display, direction_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            # Target lost
            status_text = "TARGET LOST"
            cv2.putText(frame_for_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show FPS
        if Config.SHOW_FPS:
            current_time = time.time()
            if current_time - self.last_time > 0:
                fps = 1.0 / (current_time - self.last_time)
                self.current_fps = fps
            self.last_time = current_time
            
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(frame_for_display, fps_text, (10, frame_for_display.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame_for_display, frame_text, (10, frame_for_display.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show ROI status and draw ROI rectangle if enabled
        if self.roi_enabled:
            roi_text = f"ROI: {self.roi_rect[2]}x{self.roi_rect[3]}"
            cv2.putText(frame_for_display, roi_text, (10, frame_for_display.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw ROI rectangle
            x, y, w, h = self.roi_rect
            cv2.rectangle(frame_for_display, (x, y), (x + w, y + h), (0, 255, 255), 1)
        else:
            roi_text = "ROI: FULL"
            cv2.putText(frame_for_display, roi_text, (10, frame_for_display.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame_for_display
    
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
        
        # Step 2: Two-stage masking (ROI-based or full-frame)
        if self.roi_enabled and self.frames_since_full_search < Config.MAX_MISSING_FRAMES:
            # Stage A: ROI-based search
            x, y, w, h = self.roi_rect
            
            # Extract ROI from BGR frame
            sub_bgr = self.bgr_frame_prealloc[y:y+h, x:x+w]
            
            # Convert ROI to HSV
            sub_hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
            
            # Compute mask for ROI only using separate method
            mask_roi = self._compute_roi_mask(sub_hsv)
            
            # Create full mask with zeros and copy ROI mask back
            self.mask_prealloc[:] = 0
            self.mask_prealloc[y:y+h, x:x+w] = mask_roi
            
            mask = self.mask_prealloc
        else:
            # Stage A: Full-frame search
            cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)
            mask = self.compute_mask(self.hsv_frame)
            
            # Reset ROI search counter after full search
            self.frames_since_full_search = 0
            self.roi_enabled = False
        
        # Step 4: Find best candidate
        candidate = self.find_candidate(mask)
        
        # Step 5: Update tracking
        self.update_track(candidate)
        
        # Step 6: Get state for output
        cx, cy, radius, locked, bbox = self.get_state()
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        
        # Step 7: Output state (CSV format)
        if Config.OUTPUT_CSV:
            print(f"{self.frame_count},{cx},{cy},{radius},{int(locked)},{bbox_x},{bbox_y},{bbox_w},{bbox_h}")
        
        # Step 7.5: Log frame data for result logging
        if self.result_logger:
            self.result_logger.log_frame(
                self.frame_count, cx, cy, radius, locked, bbox, 
                self.lock_duration, self.current_fps
            )
        
        # Step 8: Annotate frame for visualization (in-place)
        frame_for_display = self.annotate(frame_proc)
        
        # Store mask for debug visualization
        self._last_mask = mask
        
        return frame_for_display
    
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
    parser.add_argument('--no-logging', action='store_true', help='Disable result logging')
    parser.add_argument('--threaded', action='store_true', help='Use threaded frame capture for better performance')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.source:
        Config.VIDEO_SOURCE = args.source
    elif args.camera:
        Config.VIDEO_SOURCE = 0
    
    if args.debug:
        Config.DEBUG = True
    
    if args.csv:
        Config.OUTPUT_CSV = True
        print("frame,cx,cy,radius,locked,bbox_x,bbox_y,bbox_w,bbox_h")  # CSV header
    
    if args.no_logging:
        Config.ENABLE_RESULT_LOGGING = False
    
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
    
    # Step 2: Initialize result logger
    result_logger = None
    if Config.ENABLE_RESULT_LOGGING:
        video_source_str = str(Config.VIDEO_SOURCE)
        result_logger = ResultLogger(video_source_str)
        print(f"Result logging enabled. Results will be saved to: {Config.LOG_DIRECTORY}/")
    
    # Step 3: Instantiate detector
    detector = TargetTracker(result_logger, Config.VIDEO_SOURCE)
    
    # Step 4: Create window if not headless
    if not args.no_display:
        cv2.namedWindow("Target Tracker - Main View", cv2.WINDOW_NORMAL)
        if Config.DEBUG and Config.SHOW_MASK:
            cv2.namedWindow("Target Tracker - Color Mask", cv2.WINDOW_NORMAL)
    
    print("Target Tracker started. Main window shows tracking with bounding box.")
    print("Controls: 'q'=quit, 's'=save frame, 'm'=toggle mask view, 'b'=toggle bounding box, 'c'=toggle color format")
    print(f"Target: Red objects with area >= {Config.MIN_AREA} pixels")
    print(f"Frame size: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"Bounding box display: {'ON' if Config.SHOW_BOUNDING_BOX else 'OFF'}")
    print(f"Result logging: {'ON' if Config.ENABLE_RESULT_LOGGING else 'OFF'}")
    print(f"Threaded capture: {'ON' if args.threaded else 'OFF'}")
    if not args.no_display:
        print("Look for window: 'Target Tracker - Main View' (this shows the red object with green bounding box)")
    
    # Step 5: Setup frame capture (threaded or direct)
    frame_queue = None
    capture_thread = None
    
    if args.threaded:
        # Setup threaded frame capture
        frame_queue = Queue(maxsize=2)
        
        def capture_frames():
            """Thread function for frame capture"""
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Skip frame if queue is full (drop frames to maintain real-time)
                if not frame_queue.full():
                    frame_queue.put(frame)
        
        capture_thread = Thread(target=capture_frames, daemon=True)
        capture_thread.start()
        print("Threaded frame capture started")
    
    # Step 6: Main processing loop
    try:
        while True:
            if args.threaded:
                # Threaded frame capture
                if frame_queue.empty():
                    time.sleep(0.001)  # Small delay to prevent busy waiting
                    continue
                frame = frame_queue.get()
            else:
                # Direct frame capture
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video or failed to read frame (frame {detector.frame_count})")
                    break
            
            # Process frame through the complete pipeline
            frame_for_display = detector.process_frame(frame)
            
            # Display results (if not headless)
            if not args.no_display:
                # Always show main tracking view
                cv2.imshow('Target Tracker - Main View', frame_for_display)
                
                # Show mask in debug mode if enabled
                if Config.DEBUG and Config.SHOW_MASK:
                    mask = detector.get_debug_mask()
                    if mask is not None:
                        cv2.imshow('Target Tracker - Color Mask', mask)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame (copy only when saving)
                    filename = f"frame_{detector.frame_count:06d}.jpg"
                    to_save = frame_for_display.copy()
                    cv2.imwrite(filename, to_save)
                    print(f"Saved frame: {filename}")
                elif key == ord('m'):
                    # Toggle mask view
                    Config.SHOW_MASK = not Config.SHOW_MASK
                    if Config.SHOW_MASK and Config.DEBUG:
                        cv2.namedWindow("Target Tracker - Color Mask", cv2.WINDOW_NORMAL)
                    elif not Config.SHOW_MASK:
                        cv2.destroyWindow('Target Tracker - Color Mask')
                elif key == ord('d'):
                    # Toggle debug mode
                    Config.DEBUG = not Config.DEBUG
                    print(f"Debug mode: {'ON' if Config.DEBUG else 'OFF'}")
                elif key == ord('b'):
                    # Toggle bounding box display
                    Config.SHOW_BOUNDING_BOX = not Config.SHOW_BOUNDING_BOX
                    print(f"Bounding box display: {'ON' if Config.SHOW_BOUNDING_BOX else 'OFF'}")
                elif key == ord('c'):
                    # Toggle color format conversion
                    if detector.is_camera_source:
                        detector.needs_rgb_to_bgr_conversion = not detector.needs_rgb_to_bgr_conversion
                        detector.color_format_detected = True  # Mark as manually set
                        format_str = "RGB->BGR" if detector.needs_rgb_to_bgr_conversion else "BGR (no conversion)"
                        print(f"🔴 Color format manually set to: {format_str}")
                        print(f"💡 If this fixes the issue, you can set Config.FORCE_COLOR_FORMAT = {detector.needs_rgb_to_bgr_conversion} to make it permanent")
                    else:
                        print("Color format toggle only available for camera sources")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Step 7: Finalize result logging
        if result_logger:
            result_logger.finalize_session(detector.current_fps)
        
        # Step 8: Cleanup
        cap.release()
        if capture_thread and capture_thread.is_alive():
            print("Stopping capture thread...")
            # Thread will stop when cap.read() fails after cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Target Tracker stopped")

if __name__ == "__main__":
    main() 