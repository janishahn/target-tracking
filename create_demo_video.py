"""
Create a demo video with a moving red circle for testing the target tracker.
"""

import cv2
import numpy as np
import math

def create_demo_video(filename="demo.mp4", duration=10, fps=30):
    """
    Create a demo video with a moving red circle.
    
    Args:
        filename: Output video filename
        duration: Video duration in seconds
        fps: Frames per second
    """
    # Video properties
    width, height = 640, 480
    total_frames = duration * fps
    
    # Circle properties
    circle_radius = 25
    circle_color = (0, 0, 255)  # Red in BGR
    circle_thickness = -1  # Filled circle
    
    # Initialize video writer with more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Changed from mp4v to XVID
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Warning: Could not open video writer with XVID, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
    if not out.isOpened():
        print("Error: Could not initialize video writer")
        return
    
    print(f"Creating demo video: {filename}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}")
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate circle position (circular motion)
        t = frame_num / fps  # Time in seconds
        
        # Circular motion parameters
        center_x = width // 2
        center_y = height // 2
        orbit_radius = min(width, height) // 3
        
        # Calculate position
        angle = 2 * math.pi * t / 5  # Complete circle every 5 seconds
        x = int(center_x + orbit_radius * math.cos(angle))
        y = int(center_y + orbit_radius * math.sin(angle))
        
        # Ensure circle stays within bounds
        x = max(circle_radius, min(width - circle_radius, x))
        y = max(circle_radius, min(height - circle_radius, y))
        
        # Draw the red circle
        cv2.circle(frame, (x, y), circle_radius, circle_color, circle_thickness)
        
        # Add some noise/distraction (optional)
        if frame_num % 60 == 0:  # Every 2 seconds
            # Add a small blue circle as distraction
            noise_x = np.random.randint(50, width - 50)
            noise_y = np.random.randint(50, height - 50)
            cv2.circle(frame, (noise_x, noise_y), 10, (255, 0, 0), -1)
        
        # Add frame number for debugging
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Progress indicator
        if frame_num % (total_frames // 10) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    # Release video writer
    out.release()
    print(f"Demo video created successfully: {filename}")
    print(f"You can now test with: python target_tracker.py --source {filename} --debug")

def create_complex_demo_video(filename="demo_complex.mp4", duration=15, fps=30):
    """
    Create a more complex demo video with multiple scenarios.
    """
    width, height = 640, 480
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Changed from mp4v to XVID
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Warning: Could not open video writer with XVID, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
    if not out.isOpened():
        print("Error: Could not initialize video writer")
        return
    
    print(f"Creating complex demo video: {filename}")
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = frame_num / fps
        
        # Different scenarios based on time
        if t < 5:
            # Scenario 1: Simple circular motion
            center_x = width // 2
            center_y = height // 2
            orbit_radius = 100
            angle = 2 * math.pi * t / 3
            x = int(center_x + orbit_radius * math.cos(angle))
            y = int(center_y + orbit_radius * math.sin(angle))
            cv2.circle(frame, (x, y), 25, (0, 0, 255), -1)
            
        elif t < 8:
            # Scenario 2: Linear motion with size change
            x = int(50 + (width - 100) * ((t - 5) / 3))
            y = height // 2
            radius = int(15 + 15 * math.sin(2 * math.pi * (t - 5)))
            cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)
            
        elif t < 12:
            # Scenario 3: Target disappears and reappears
            if int(t * 2) % 2 == 0:  # Blink every 0.5 seconds
                x = int(width * 0.3 + width * 0.4 * ((t - 8) / 4))
                y = int(height * 0.7)
                cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
            
        else:
            # Scenario 4: Multiple red objects (test largest selection)
            # Large target
            x1 = int(width * 0.3)
            y1 = int(height * 0.3)
            cv2.circle(frame, (x1, y1), 30, (0, 0, 255), -1)
            
            # Smaller targets
            x2 = int(width * 0.7)
            y2 = int(height * 0.3)
            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), -1)
            
            x3 = int(width * 0.5)
            y3 = int(height * 0.7)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), -1)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % (total_frames // 10) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"Complex demo video created: {filename}")

def main():
    """Create demo videos"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create demo videos for target tracking')
    parser.add_argument('--simple', action='store_true', help='Create simple demo video')
    parser.add_argument('--complex', action='store_true', help='Create complex demo video')
    parser.add_argument('--both', action='store_true', help='Create both demo videos')
    parser.add_argument('--duration', type=int, default=10, help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    if args.both or args.simple or (not args.complex and not args.simple):
        create_demo_video("demo.mp4", args.duration, args.fps)
    
    if args.both or args.complex:
        create_complex_demo_video("demo_complex.mp4", args.duration, args.fps)
    
    print("\nDemo videos created successfully!")
    print("Test with:")
    print("  python target_tracker.py --source demo.mp4 --debug")
    print("  python target_tracker.py --source demo_complex.mp4 --debug")

if __name__ == "__main__":
    main() 