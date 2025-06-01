"""
Create advanced demo videos with complex scenarios for testing the target tracker.
Includes partial occlusion, diverse colors, moving obstacles, and challenging tracking scenarios.
"""

import cv2
import numpy as np
import math
import random

def create_advanced_demo_video(filename="demo.mp4", duration=20, fps=30):
    """
    Create an advanced demo video with challenging tracking scenarios.
    
    Scenarios include:
    - Partial occlusion by moving obstacles
    - Diverse background colors and distractors
    - Target size/shape variations
    - Multiple red objects (largest selection test)
    - Temporary disappearance and reappearance
    - Cluttered backgrounds
    
    Args:
        filename: Output video filename
        duration: Video duration in seconds
        fps: Frames per second
    """
    # Video properties
    width, height = 1280, 960
    total_frames = duration * fps
    
    # Target properties
    target_color = (0, 0, 255)  # Red in BGR
    base_radius = 25
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Warning: Could not open video writer with XVID, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
    if not out.isOpened():
        print("Error: Could not initialize video writer")
        return
    
    print(f"Creating advanced demo video: {filename}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}")
    print("Scenarios: Occlusion, diverse colors, size changes, multiple targets")
    
    # Initialize moving obstacles
    obstacles = []
    for i in range(3):
        obstacles.append({
            'x': random.randint(50, width-50),
            'y': random.randint(50, height-50),
            'vx': random.uniform(-2, 2),
            'vy': random.uniform(-2, 2),
            'size': random.randint(30, 60),
            'color': (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        })
    
    for frame_num in range(total_frames):
        t = frame_num / fps  # Time in seconds
        
        # Create dynamic background
        if t < 5:
            # Clean dark background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        elif t < 10:
            # Gradually introduce colorful background
            intensity = int(50 * ((t - 5) / 5))
            frame = np.full((height, width, 3), intensity, dtype=np.uint8)
            # Add random noise
            noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
        else:
            # Complex textured background
            frame = np.random.randint(20, 80, (height, width, 3), dtype=np.uint8)
            # Add some patterns
            for y in range(0, height, 40):
                cv2.line(frame, (0, y), (width, y), (100, 100, 100), 1)
            for x in range(0, width, 40):
                cv2.line(frame, (x, 0), (x, height), (100, 100, 100), 1)
        
        # Add diverse colored distractors
        if t > 2:
            num_distractors = int(3 + 2 * (t / duration))
            for i in range(num_distractors):
                dist_x = random.randint(20, width-20)
                dist_y = random.randint(20, height-20)
                dist_size = random.randint(5, 15)
                # Various colors except red
                colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                dist_color = random.choice(colors)
                cv2.circle(frame, (dist_x, dist_y), dist_size, dist_color, -1)
        
        # Calculate main target position and properties
        scenario_time = t % 16  # 16-second cycle of scenarios
        
        if scenario_time < 4:
            # Scenario 1: Simple circular motion with size variation
            center_x = width // 2
            center_y = height // 2
            orbit_radius = 80
            angle = 2 * math.pi * scenario_time / 3
            target_x = int(center_x + orbit_radius * math.cos(angle))
            target_y = int(center_y + orbit_radius * math.sin(angle))
            target_radius = int(base_radius + 10 * math.sin(4 * angle))
            
        elif scenario_time < 8:
            # Scenario 2: Linear motion with obstacles
            progress = (scenario_time - 4) / 4
            target_x = int(50 + (width - 100) * progress)
            target_y = int(height // 2 + 50 * math.sin(2 * math.pi * progress))
            target_radius = base_radius
            
        elif scenario_time < 12:
            # Scenario 3: Erratic motion with partial occlusion
            base_x = width // 2
            base_y = height // 2
            noise_factor = (scenario_time - 8) / 4
            target_x = int(base_x + 100 * math.sin(2 * math.pi * scenario_time) * noise_factor)
            target_y = int(base_y + 80 * math.cos(3 * math.pi * scenario_time) * noise_factor)
            target_radius = int(base_radius + 5 * math.sin(6 * math.pi * scenario_time))
            
        else:
            # Scenario 4: Multiple red targets (test largest selection)
            # Draw multiple red circles, tracker should pick the largest
            target_x = int(width * 0.3 + 100 * math.sin(math.pi * scenario_time))
            target_y = int(height * 0.5)
            target_radius = 35  # Largest target
            
            # Smaller red distractors
            small_targets = [
                (int(width * 0.7), int(height * 0.3), 15),
                (int(width * 0.8), int(height * 0.7), 12),
                (int(width * 0.2), int(height * 0.8), 18)
            ]
            for sx, sy, sr in small_targets:
                cv2.circle(frame, (sx, sy), sr, target_color, -1)
        
        # Update and draw moving obstacles
        for obstacle in obstacles:
            # Update position
            obstacle['x'] += obstacle['vx']
            obstacle['y'] += obstacle['vy']
            
            # Bounce off walls
            if obstacle['x'] <= obstacle['size'] or obstacle['x'] >= width - obstacle['size']:
                obstacle['vx'] *= -1
            if obstacle['y'] <= obstacle['size'] or obstacle['y'] >= height - obstacle['size']:
                obstacle['vy'] *= -1
            
            # Keep within bounds
            obstacle['x'] = max(obstacle['size'], min(width - obstacle['size'], obstacle['x']))
            obstacle['y'] = max(obstacle['size'], min(height - obstacle['size'], obstacle['y']))
            
            # Draw obstacle (only after 6 seconds)
            if t > 6:
                cv2.circle(frame, (int(obstacle['x']), int(obstacle['y'])), 
                          obstacle['size'], obstacle['color'], -1)
                # Add some transparency effect by drawing smaller circles
                cv2.circle(frame, (int(obstacle['x']), int(obstacle['y'])), 
                          obstacle['size'] - 5, (100, 100, 100), 2)
        
        # Draw main target (ensure it's visible but can be partially occluded)
        target_x = max(target_radius, min(width - target_radius, target_x))
        target_y = max(target_radius, min(height - target_radius, target_y))
        
        # Intermittent visibility (target disappears briefly)
        visible = True
        if 14 < t < 16:
            # Target blinks during this period
            visible = (int(t * 4) % 2 == 0)
        
        if visible:
            # Draw main target with slight transparency effect for realism
            cv2.circle(frame, (target_x, target_y), target_radius, target_color, -1)
            # Add a subtle white outline for better visibility
            cv2.circle(frame, (target_x, target_y), target_radius, (255, 255, 255), 1)
        
        # Add some motion blur effect occasionally
        if frame_num % 90 == 0 and t > 10:  # Every 3 seconds after t=10
            kernel = np.ones((3, 3), np.float32) / 9
            frame = cv2.filter2D(frame, -1, kernel)
        
        # Add frame information
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        scenario_names = ["Circular+Size", "Linear+Obstacles", "Erratic+Occlusion", "Multiple Targets"]
        scenario_idx = int(scenario_time // 4)
        cv2.putText(frame, f"Scenario: {scenario_names[scenario_idx]}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not visible:
            cv2.putText(frame, "TARGET HIDDEN", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Progress indicator
        if frame_num % (total_frames // 20) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    # Release video writer
    out.release()
    print(f"Advanced demo video created successfully: {filename}")
    print(f"Test with: python target_tracker.py --source {filename} --debug")

def create_occlusion_test_video(filename="demo_occlusion.mp4", duration=15, fps=30):
    """
    Create a specialized video focusing on occlusion scenarios.
    """
    width, height = 1280, 960
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
    if not out.isOpened():
        print("Error: Could not initialize video writer")
        return
    
    print(f"Creating occlusion test video: {filename}")
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = frame_num / fps
        
        # Moving target
        target_x = int(100 + 400 * (t / duration))
        target_y = int(height // 2 + 50 * math.sin(2 * math.pi * t / 3))
        target_radius = 25
        
        # Draw target
        cv2.circle(frame, (target_x, target_y), target_radius, (0, 0, 255), -1)
        
        # Moving occluder (vertical bar)
        occluder_x = int(200 + 200 * math.sin(math.pi * t / 4))
        cv2.rectangle(frame, (occluder_x, 0), (occluder_x + 40, height), (128, 128, 128), -1)
        
        # Static occluders
        cv2.rectangle(frame, (300, 150), (340, 250), (100, 100, 100), -1)
        cv2.rectangle(frame, (450, 200), (490, 350), (80, 80, 80), -1)
        
        # Add partial occluders (semi-transparent effect)
        overlay = frame.copy()
        cv2.rectangle(overlay, (150, 100), (250, 300), (60, 60, 60), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"Occlusion Test - Time: {t:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % (total_frames // 10) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"Occlusion test video created: {filename}")

def main():
    """Create demo videos with various complexity levels"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create advanced demo videos for target tracking')
    parser.add_argument('--advanced', action='store_true', help='Create advanced demo video (default)')
    parser.add_argument('--occlusion', action='store_true', help='Create occlusion-focused test video')
    parser.add_argument('--all', action='store_true', help='Create all demo videos')
    parser.add_argument('--duration', type=int, default=20, help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    if args.all or args.advanced or (not args.occlusion and not args.advanced):
        create_advanced_demo_video("demo.mp4", args.duration, args.fps)
    
    if args.all or args.occlusion:
        create_occlusion_test_video("demo_occlusion.mp4", 15, args.fps)
    
    print("\nAdvanced demo videos created successfully!")
    print("\nRecommended test commands:")
    print("  Basic test:     python target_tracker.py --source demo.mp4 --debug")
    print("  Occlusion test: python target_tracker.py --source demo_occlusion.mp4 --debug")
    print("  Performance:    python performance_test.py")
    print("  Threaded:       python target_tracker.py --source demo.mp4 --threaded --debug")

if __name__ == "__main__":
    main() 