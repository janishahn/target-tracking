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
    SHOW_BOUNDING_BOX = True    # Show bounding box instead of just circle
    
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
    
    def __init__(self, result_logger: Optional[ResultLogger] = None):
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
        
        # Result logging
        self.result_logger = result_logger
        
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
    
    def find_candidate(self, mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], int, float, Tuple[int, int, int, int], np.ndarray]]:
        """
        Find the best target candidate from binary mask.
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            
        Returns:
            Best candidate as ((x, y), radius, area, bbox, contour) or None if no valid candidate
            bbox format: (x, y, w, h) - top-left corner and dimensions
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
            
            # Get bounding box
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
            
            # Add to candidates list
            candidates.append((cnt, area, circularity, (int(x), int(y), radius), (bbox_x, bbox_y, bbox_w, bbox_h)))
        
        if not candidates:
            return None
        
        # Sort by area (descending) and pick the largest
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = candidates[0]
        
        # Return (centroid, radius, area, bbox, contour)
        contour, area, _, (x, y, radius), bbox = best_candidate
        return ((x, y), radius, area, bbox, contour)
    
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
                else:
                    # Ignore this detection, increment missing frames
                    self.missing_frames += 1
                    if self.missing_frames > Config.MAX_MISSING_FRAMES:
                        self.locked = False
                        self.lock_duration = 0
                        self.last_centroid = None
                        self.smoothed_centroid = None
                        self.last_bbox = None
                        self.smoothed_bbox = None
                        self.last_contour = None
        else:
            # No candidate this frame
            self.missing_frames += 1
            if self.missing_frames > Config.MAX_MISSING_FRAMES:
                self.locked = False
                self.lock_duration = 0
                self.last_centroid = None
                self.smoothed_centroid = None
                self.last_bbox = None
                self.smoothed_bbox = None
                self.last_contour = None
    
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
    
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """
        Add visualization overlay to frame.
        
        Args:
            frame: BGR frame to annotate
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        cx, cy, radius, locked, bbox = self.get_state()
        
        if locked:
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            
            if Config.SHOW_BOUNDING_BOX:
                # Draw bounding box
                cv2.rectangle(vis_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 2)
                
                # Draw target circle (optional, for reference)
                cv2.circle(vis_frame, (cx, cy), radius, (0, 255, 0), 1)
            else:
                # Draw target circle only
                cv2.circle(vis_frame, (cx, cy), radius, (0, 255, 0), 2)
            
            # Draw crosshairs
            cv2.line(vis_frame, (cx-10, cy), (cx+10, cy), (0, 255, 0), 2)
            cv2.line(vis_frame, (cx, cy-10), (cx, cy+10), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, (cx, cy), 2, (0, 255, 0), -1)
            
            # Add bounding box coordinates at top-left corner of the box (only if showing bounding box)
            if Config.SHOW_BOUNDING_BOX:
                bbox_coord_text = f"({bbox_x},{bbox_y})"
                # Position text slightly above the bounding box
                text_x = bbox_x
                text_y = max(bbox_y - 5, 15)  # Ensure text doesn't go off screen
                
                # Add background rectangle for better text visibility
                text_size = cv2.getTextSize(bbox_coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(vis_frame, (text_x, text_y - text_size[1] - 2), 
                             (text_x + text_size[0] + 4, text_y + 2), (0, 0, 0), -1)
                
                # Draw the coordinate text
                cv2.putText(vis_frame, bbox_coord_text, (text_x + 2, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Add bounding box dimensions next to coordinates
                bbox_size_text = f"{bbox_w}x{bbox_h}"
                cv2.putText(vis_frame, bbox_size_text, (text_x + 2, text_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Add text info (status)
            info_text = f"LOCKED ({self.lock_duration} frames)"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show center coordinates
            coord_text = f"Center: ({cx}, {cy})"
            cv2.putText(vis_frame, coord_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show bounding box info
            bbox_info_text = f"BBox: ({bbox_x},{bbox_y}) {bbox_w}x{bbox_h}"
            cv2.putText(vis_frame, bbox_info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
    parser.add_argument('--no-logging', action='store_true', help='Disable result logging')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.source:
        Config.VIDEO_SOURCE = args.source
    elif args.camera:
        Config.VIDEO_SOURCE = 0
    
    if args.debug:
        Config.DEBUG = True
        # Don't automatically show mask in debug mode - let user toggle with 'm' key
    
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
    detector = TargetTracker(result_logger)
    
    # Step 4: Create window if not headless
    if not args.no_display:
        cv2.namedWindow("Target Tracker - Main View", cv2.WINDOW_NORMAL)
        if Config.DEBUG and Config.SHOW_MASK:
            cv2.namedWindow("Target Tracker - Color Mask", cv2.WINDOW_NORMAL)
    
    print("Target Tracker started. Main window shows tracking with bounding box.")
    print("Controls: 'q'=quit, 's'=save frame, 'm'=toggle mask view, 'b'=toggle bounding box")
    print(f"Target: Red objects with area >= {Config.MIN_AREA} pixels")
    print(f"Frame size: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"Bounding box display: {'ON' if Config.SHOW_BOUNDING_BOX else 'OFF'}")
    print(f"Result logging: {'ON' if Config.ENABLE_RESULT_LOGGING else 'OFF'}")
    if not args.no_display:
        print("Look for window: 'Target Tracker - Main View' (this shows the red object with green bounding box)")
    
    # Step 5: Main processing loop
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
                # Always show main tracking view
                cv2.imshow('Target Tracker - Main View', vis_frame)
                
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
                    # Save current frame
                    filename = f"frame_{detector.frame_count:06d}.jpg"
                    cv2.imwrite(filename, vis_frame)
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
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Step 6: Finalize result logging
        if result_logger:
            result_logger.finalize_session(detector.current_fps)
        
        # Step 7: Cleanup
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Target Tracker stopped")

if __name__ == "__main__":
    main() 