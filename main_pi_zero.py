#!/usr/bin/env python3
"""
Raspberry Pi Zero 2W Headless Target Tracking with Periodic Snapshots
Optimized for headless operation with performance monitoring and snapshot capture
Supports both live camera feed and MP4 file processing
"""

import time
import threading
import os
import json
import argparse
from queue import Queue
from datetime import datetime
from typing import Dict, Any, Optional

import cv2
from picamera2 import Picamera2
from target_tracker import TargetTracker, Config

class HeadlessPerformanceTracker:
    """
    Performance tracking for headless Pi Zero operation.
    Tracks FPS, detection rates, and lock performance without creating markdown files.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.session_start = datetime.now()
        
        # Performance metrics
        self.total_frames = 0
        self.frames_with_detection = 0
        self.frames_locked = 0
        self.lock_sessions = []
        self.current_lock_start = None
        self.current_lock_duration = 0
        
        # FPS tracking
        self.fps_samples = []
        self.last_frame_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Performance tracking initialized. Output: {output_dir}")
    
    def update_frame_metrics(self, has_detection: bool, is_locked: bool, lock_duration: int):
        """Update frame-level performance metrics"""
        current_time = time.time()
        
        # Calculate instantaneous FPS
        if self.total_frames > 0:
            frame_fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_samples.append(frame_fps)
            
            # Keep only last 100 samples for rolling average
            if len(self.fps_samples) > 100:
                self.fps_samples.pop(0)
        
        self.last_frame_time = current_time
        self.total_frames += 1
        
        if has_detection:
            self.frames_with_detection += 1
        
        if is_locked:
            self.frames_locked += 1
            self.current_lock_duration = lock_duration
            
            # Track lock session start
            if self.current_lock_start is None:
                self.current_lock_start = self.total_frames
        else:
            # End current lock session if it was active
            if self.current_lock_start is not None:
                session_data = {
                    'start_frame': self.current_lock_start,
                    'end_frame': self.total_frames - 1,
                    'duration': self.current_lock_duration
                }
                self.lock_sessions.append(session_data)
                self.current_lock_start = None
                self.current_lock_duration = 0
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0
        detection_rate = (self.frames_with_detection / self.total_frames * 100) if self.total_frames > 0 else 0
        lock_rate = (self.frames_locked / self.total_frames * 100) if self.total_frames > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'avg_fps': avg_fps,
            'detection_rate': detection_rate,
            'lock_rate': lock_rate,
            'lock_sessions': len(self.lock_sessions),
            'current_lock_duration': self.current_lock_duration
        }
    
    def save_final_report(self):
        """Save final performance report as JSON"""
        # Close any ongoing lock session
        if self.current_lock_start is not None:
            session_data = {
                'start_frame': self.current_lock_start,
                'end_frame': self.total_frames,
                'duration': self.current_lock_duration
            }
            self.lock_sessions.append(session_data)
        
        stats = self.get_current_stats()
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        final_report = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': session_duration,
                'device': 'Raspberry Pi Zero 2W'
            },
            'performance_metrics': {
                'total_frames_processed': self.total_frames,
                'average_fps': stats['avg_fps'],
                'detection_rate_percent': stats['detection_rate'],
                'lock_rate_percent': stats['lock_rate'],
                'total_lock_sessions': len(self.lock_sessions),
                'frames_with_detection': self.frames_with_detection,
                'frames_locked': self.frames_locked
            },
            'lock_sessions': self.lock_sessions,
            'configuration': {
                'frame_width': Config.FRAME_WIDTH,
                'frame_height': Config.FRAME_HEIGHT,
                'min_area': Config.MIN_AREA,
                'min_circularity': Config.MIN_CIRCULARITY,
                'smoothing_alpha': Config.SMOOTHING_ALPHA
            }
        }
        
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"{timestamp}_performance_report.json")
        
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"Final performance report saved: {report_file}")
        return final_report

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

def save_snapshot(frame: any, output_dir: str, frame_number: int, tracker_state: tuple, is_camera_feed: bool = False, debug_mask: any = None) -> str:
    """
    Save annotated frame snapshot with tracking information
    
    Args:
        frame: Annotated frame from tracker (always in BGR format after processing)
        output_dir: Directory to save snapshots
        frame_number: Current frame number
        tracker_state: (cx, cy, radius, locked, bbox) from tracker
        is_camera_feed: True if frame came from live camera
        debug_mask: Optional debug mask to save alongside the frame
        
    Returns:
        Filename of saved snapshot
    """
    timestamp = datetime.now().strftime("%H%M%S")
    cx, cy, radius, locked, bbox = tracker_state
    status = "LOCKED" if locked else "LOST"
    
    filename = f"snapshot_f{frame_number:06d}_{timestamp}_{status}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # The frame from tracker is always in BGR format after processing
    # cv2.imwrite() expects BGR format, so we save directly
    # The color inversion issue was due to the tracker's automatic RGB->BGR conversion
    # which is now handled correctly in the tracker itself
    cv2.imwrite(filepath, frame)
    
    # Save debug mask if provided
    if debug_mask is not None:
        mask_filename = f"mask_f{frame_number:06d}_{timestamp}_{status}.jpg"
        mask_filepath = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_filepath, debug_mask)
        print(f"Debug mask saved: {mask_filename}")
    
    return filename

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Raspberry Pi Zero 2W Headless Target Tracker')
    parser.add_argument('--source', type=str, help='MP4 video file to process instead of live camera')
    return parser.parse_args()

def main():
    """
    Main headless application loop for Pi Zero 2W
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("RASPBERRY PI ZERO 2W HEADLESS TARGET TRACKER")
    print("=" * 60)
    
    if args.source:
        print(f"Processing video file: {args.source}")
        if not os.path.exists(args.source):
            print(f"Error: Video file '{args.source}' not found!")
            return
    else:
        print("Using live camera feed...")
    
    print("Headless operation with periodic snapshots...")
    
    # Configuration for headless operation
    output_dir = "pi_zero_output"
    snapshot_interval = 10  # Save snapshot every N frames
    status_interval = 150   # Print status every N frames
    
    # Enable diagnostic mode for detection debugging
    diagnostic_mode = True
    
    # Disable all GUI-related config options
    Config.SHOW_FPS = False
    Config.SHOW_MASK = False
    Config.DEBUG = False
    Config.ENABLE_RESULT_LOGGING = False  # Disable the built-in logging
    
    # Diagnostic: Relax detection parameters for better detection
    if diagnostic_mode:
        print("DIAGNOSTIC MODE ENABLED - Relaxing detection parameters")
        Config.MIN_AREA = 50  # Reduce from 200 to 50
        Config.MIN_CIRCULARITY = 0.3  # Reduce from 0.6 to 0.3
        Config.HSV_LOWER1 = (0, 50, 50)    # More permissive red range
        Config.HSV_UPPER1 = (15, 255, 255)
        Config.HSV_LOWER2 = (155, 50, 50)  # More permissive red range  
        Config.HSV_UPPER2 = (180, 255, 255)
        snapshot_interval = 10  # More frequent snapshots for debugging
    
    print(f"Output directory: {output_dir}")
    print(f"Snapshot interval: every {snapshot_interval} frames")
    print(f"Status interval: every {status_interval} frames")
    
    # Step 1: Initialize video source
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
    
    # Step 2: Setup frame queue and capture thread
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
    
    # Step 3: Initialize tracking components
    video_source = args.source if args.source else 0  # 0 for camera, file path for video
    tracker = TargetTracker(result_logger=None, video_source=video_source)  # No result logger
    performance_tracker = HeadlessPerformanceTracker(output_dir)
    
    print("\nHeadless tracking started!")
    print(f"Camera FOV: 640x480 (double resolution for larger field of view)")
    print(f"Processing resolution: 320x240 (maintains Pi Zero workload)")
    print(f"Target: Red objects, min area: {Config.MIN_AREA} pixels")
    if diagnostic_mode:
        print(f"DIAGNOSTIC MODE: Relaxed parameters - Area≥{Config.MIN_AREA}, Circularity≥{Config.MIN_CIRCULARITY}")
        print(f"HSV ranges: {Config.HSV_LOWER1}-{Config.HSV_UPPER1} and {Config.HSV_LOWER2}-{Config.HSV_UPPER2}")
    print("Press Ctrl+C to stop...")
    
    try:
        # Step 4: Main processing loop
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
            
            # Get tracking state
            cx, cy, radius, locked, bbox = tracker.get_state()
            has_detection = (cx != -1 and cy != -1)
            
            # Update performance metrics
            performance_tracker.update_frame_metrics(
                has_detection, locked, tracker.lock_duration
            )
            
            # Save periodic snapshots with debug information
            if tracker.frame_count % snapshot_interval == 0:
                debug_mask = tracker.get_debug_mask() if diagnostic_mode else None
                filename = save_snapshot(
                    annotated_frame, output_dir, tracker.frame_count, 
                    (cx, cy, radius, locked, bbox), is_camera_feed=not args.source,
                    debug_mask=debug_mask
                )
                print(f"Snapshot saved: {filename}")
                
                # Diagnostic: Print detailed detection info
                if diagnostic_mode:
                    mask_stats = cv2.countNonZero(debug_mask) if debug_mask is not None else 0
                    print(f"  Debug info - Mask pixels: {mask_stats}, Detection: {has_detection}")
                    print(f"  HSV ranges: {Config.HSV_LOWER1}-{Config.HSV_UPPER1}, {Config.HSV_LOWER2}-{Config.HSV_UPPER2}")
                    print(f"  Min area: {Config.MIN_AREA}, Min circularity: {Config.MIN_CIRCULARITY}")
            
            # Print status periodically
            if tracker.frame_count % status_interval == 0:
                stats = performance_tracker.get_current_stats()
                print(f"\n--- Frame {tracker.frame_count} Status ---")
                print(f"FPS: {stats['avg_fps']:.1f}")
                print(f"Detection Rate: {stats['detection_rate']:.1f}%")
                print(f"Lock Rate: {stats['lock_rate']:.1f}%")
                print(f"Lock Sessions: {stats['lock_sessions']}")
                if locked:
                    print(f"Current Lock: {stats['current_lock_duration']} frames")
                    print(f"Target: ({cx}, {cy}) radius={radius}")
                else:
                    print("Status: TARGET LOST")
                print("-" * 35)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Step 5: Cleanup and final reporting
        print("Shutting down headless tracker...")
        
        # Save final performance report
        final_report = performance_tracker.save_final_report()
        
        # Print final summary
        print("\n" + "=" * 50)
        print("FINAL PERFORMANCE SUMMARY")
        print("=" * 50)
        metrics = final_report['performance_metrics']
        print(f"Total Frames: {metrics['total_frames_processed']}")
        print(f"Average FPS: {metrics['average_fps']:.1f}")
        print(f"Detection Rate: {metrics['detection_rate_percent']:.1f}%")
        print(f"Lock Rate: {metrics['lock_rate_percent']:.1f}%")
        print(f"Lock Sessions: {metrics['total_lock_sessions']}")
        print(f"Session Duration: {final_report['session_info']['duration_seconds']:.1f}s")
        print(f"Output Directory: {output_dir}")
        
        # Stop camera if it was used
        if picam2:
            picam2.stop()
            print("Camera stopped")
        
        print("Headless tracker stopped successfully!")

if __name__ == "__main__":
    main() 