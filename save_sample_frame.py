"""
Save a sample frame with tracking visualization
"""

import cv2
from target_tracker import TargetTracker

def save_sample_frame():
    # Test with demo video and save a frame
    cap = cv2.VideoCapture('demo.mp4')
    tracker = TargetTracker()

    # Read a few frames to get tracking started
    for i in range(50):
        ret, frame = cap.read()
        if ret:
            vis_frame = tracker.process_frame(frame)

    # Save the visualization
    cv2.imwrite('tracking_sample.jpg', vis_frame)
    print('Saved tracking visualization to tracking_sample.jpg')
    print(f'Frame {tracker.frame_count}: Target locked = {tracker.locked}')
    if tracker.locked:
        cx, cy, radius, locked = tracker.get_state()
        print(f'Target position: ({cx}, {cy}), radius: {radius}')

    cap.release()

if __name__ == "__main__":
    save_sample_frame() 