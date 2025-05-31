#!/usr/bin/env python3
"""
Stream Viewer for Pi Zero Target Tracker
View the live MJPEG stream from Raspberry Pi in a local OpenCV window
"""

import cv2
import argparse
import time
import sys
from urllib.request import urlopen
import numpy as np

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='View Pi Zero Target Tracker Stream')
    parser.add_argument('pi_ip', type=str, help='Raspberry Pi IP address')
    parser.add_argument('--port', type=int, default=8080, help='Streaming port (default: 8080)')
    parser.add_argument('--save-frames', action='store_true', help='Save frames to disk')
    return parser.parse_args()

def main():
    """Main viewer application"""
    args = parse_arguments()
    
    stream_url = f"http://{args.pi_ip}:{args.port}/stream"
    
    print("=" * 60)
    print("PI ZERO TARGET TRACKER - STREAM VIEWER")
    print("=" * 60)
    print(f"Connecting to: {stream_url}")
    print("Press 'q' to quit, 's' to save current frame")
    print("=" * 60)
    
    # Try to connect to the stream
    try:
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"Error: Could not connect to stream at {stream_url}")
            print("Make sure:")
            print("1. Pi is running the streaming script")
            print("2. Pi IP address is correct")
            print("3. Port is correct")
            print("4. Both devices are on the same network")
            return
        
        print("✅ Connected successfully!")
        print(f"🎯 Viewing live target tracking from Pi at {args.pi_ip}")
        
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Connection lost or stream ended")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate and display FPS every 30 frames
            if fps_frame_count >= 30:
                elapsed = time.time() - fps_start_time
                fps = fps_frame_count / elapsed
                print(f"Viewer FPS: {fps:.1f}")
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Add frame info overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pi IP: {args.pi_ip}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Pi Zero Target Tracker Stream', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting viewer...")
                break
            elif key == ord('s') or args.save_frames:
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pi_stream_frame_{timestamp}_{frame_count:06d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")
                
                # Reset save_frames flag if it was set
                if args.save_frames and frame_count % 30 == 0:  # Save every 30 frames
                    pass  # Continue saving
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check Pi IP address")
        print("2. Ensure Pi is running the streaming script")
        print("3. Check network connectivity")
        print("4. Try accessing the stream in a web browser first")
    
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Stream viewer closed")

if __name__ == "__main__":
    main() 