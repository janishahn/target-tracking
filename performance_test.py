"""
Performance test for target tracking system
"""

import time
import cv2
from target_tracker import TargetTracker, Config

def test_performance(video_source="demo.mp4", max_frames=100):
    """Test the performance of the target tracking system"""
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open {video_source}")
        return
    
    # Initialize tracker
    tracker = TargetTracker()
    
    print(f"Performance Test - Processing {max_frames} frames")
    print(f"Video source: {video_source}")
    print(f"Frame size: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print("-" * 50)
    
    frame_count = 0
    start_time = time.time()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame (this is where all the work happens)
        vis_frame = tracker.process_frame(frame)
        
        frame_count += 1
        
        # Print progress every 25 frames
        if frame_count % 25 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(f"Frame {frame_count:3d}: {current_fps:.1f} FPS")
    
    # Final results
    total_time = time.time() - start_time
    average_fps = frame_count / total_time
    
    print("-" * 50)
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {average_fps:.1f}")
    print(f"Target locked for {tracker.lock_duration} frames")
    
    # Performance assessment
    if average_fps >= 20:
        print("✅ Excellent performance - Ready for Pi deployment")
    elif average_fps >= 15:
        print("✅ Good performance - Should work on Pi")
    elif average_fps >= 10:
        print("⚠️  Moderate performance - May need optimization for Pi")
    else:
        print("❌ Poor performance - Needs optimization")
    
    cap.release()
    return average_fps

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance test for target tracker')
    parser.add_argument('--source', type=str, default='demo.mp4', help='Video source')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to test')
    
    args = parser.parse_args()
    
    test_performance(args.source, args.frames) 