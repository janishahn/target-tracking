#!/usr/bin/env python3
"""
Raspberry Pi Zero 2W Headless Target Tracking with Periodic Snapshots
Optimized for headless operation with performance monitoring and snapshot capture
"""

import time
import threading
import os
import json
from queue import Queue
from datetime import datetime
from typing import Dict, Any

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
    Initialize camera optimized for Pi Zero 2W headless operation
    """
    print("Initializing Raspberry Pi camera (headless mode)...")
    picam2 = Picamera2()
    
    # Configure camera for optimal Pi Zero performance
    camera_config = picam2.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(camera_config)
    picam2.start()
    
    print("Camera initialized successfully (320x240 headless)")
    return picam2

def camera_capture_thread(picam2: Picamera2, frame_queue: Queue):
    """
    Optimized capture thread for headless operation
    """
    print("Starting camera capture thread...")
    while True:
        try:
            frame_rgb = picam2.capture_array("main")
            
            if not frame_queue.full():
                frame_queue.put_nowait(frame_rgb)
                
        except Exception as e:
            print(f"Error in capture thread: {e}")
            break

def save_snapshot(frame: any, output_dir: str, frame_number: int, tracker_state: tuple) -> str:
    """
    Save annotated frame snapshot with tracking information
    
    Args:
        frame: Annotated frame from tracker
        output_dir: Directory to save snapshots
        frame_number: Current frame number
        tracker_state: (cx, cy, radius, locked, bbox) from tracker
        
    Returns:
        Filename of saved snapshot
    """
    timestamp = datetime.now().strftime("%H%M%S")
    cx, cy, radius, locked, bbox = tracker_state
    status = "LOCKED" if locked else "LOST"
    
    filename = f"snapshot_f{frame_number:06d}_{timestamp}_{status}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, frame)
    return filename

def main():
    """
    Main headless application loop for Pi Zero 2W
    """
    print("=" * 60)
    print("RASPBERRY PI ZERO 2W HEADLESS TARGET TRACKER")
    print("=" * 60)
    print("Headless operation with periodic snapshots...")
    
    # Configuration for headless operation
    output_dir = "pi_zero_output"
    snapshot_interval = 30  # Save snapshot every N frames
    status_interval = 150   # Print status every N frames
    
    # Disable all GUI-related config options
    Config.SHOW_FPS = False
    Config.SHOW_MASK = False
    Config.DEBUG = False
    Config.ENABLE_RESULT_LOGGING = False  # Disable the built-in logging
    
    print(f"Output directory: {output_dir}")
    print(f"Snapshot interval: every {snapshot_interval} frames")
    print(f"Status interval: every {status_interval} frames")
    
    # Step 1: Initialize camera
    try:
        picam2 = initialize_camera_pi_zero()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        print("Make sure the camera is properly connected and enabled.")
        return
    
    # Step 2: Setup frame queue
    frame_queue = Queue(maxsize=2)
    capture_thread = threading.Thread(
        target=camera_capture_thread, 
        args=(picam2, frame_queue), 
        daemon=True
    )
    capture_thread.start()
    
    # Step 3: Initialize tracking components
    tracker = TargetTracker(result_logger=None)  # No result logger
    performance_tracker = HeadlessPerformanceTracker(output_dir)
    
    print("\nHeadless tracking started!")
    print(f"Target: Red objects, min area: {Config.MIN_AREA} pixels")
    print("Press Ctrl+C to stop...")
    
    try:
        # Step 4: Main processing loop
        while True:
            # Get frame from capture queue
            if frame_queue.empty():
                time.sleep(0.001)
                continue
                
            frame_rgb = frame_queue.get()
            
            # Convert RGB to BGR for OpenCV processing
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
            
            # Save periodic snapshots
            if tracker.frame_count % snapshot_interval == 0:
                filename = save_snapshot(
                    annotated_frame, output_dir, tracker.frame_count, 
                    (cx, cy, radius, locked, bbox)
                )
                print(f"Snapshot saved: {filename}")
            
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
        
        # Stop camera
        picam2.stop()
        print("Camera stopped")
        
        print("Headless tracker stopped successfully!")

if __name__ == "__main__":
    main() 