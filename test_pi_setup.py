#!/usr/bin/env python3
"""
Test script to verify Pi camera integration setup
Run this to check if all dependencies and camera are working correctly
"""

import sys
import importlib
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing Python imports...")
    
    required_modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('threading', 'builtin'),
        ('queue', 'builtin'),
        ('time', 'builtin')
    ]
    
    failed_imports = []
    
    for module_name, package_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name} imported successfully")
        except ImportError as e:
            print(f"  ✗ {module_name} import failed: {e}")
            failed_imports.append((module_name, package_name))
    
    return failed_imports

def test_picamera2():
    """Test if picamera2 is available (only on Pi)"""
    print("\nTesting picamera2 (Pi camera library)...")
    
    try:
        from picamera2 import Picamera2
        print("  ✓ picamera2 imported successfully")
        
        # Try to list cameras (won't work on non-Pi systems)
        try:
            picam2 = Picamera2()
            print("  ✓ Picamera2 instance created")
            return True
        except Exception as e:
            print(f"  ⚠ Picamera2 available but camera not accessible: {e}")
            print("    This is normal if running on non-Pi hardware")
            return False
            
    except ImportError as e:
        print(f"  ✗ picamera2 import failed: {e}")
        print("    Install with: pip install picamera2")
        print("    Note: picamera2 only works on Raspberry Pi")
        return False

def test_target_tracker():
    """Test if target_tracker module can be imported"""
    print("\nTesting target_tracker module...")
    
    try:
        from target_tracker import TargetTracker, ResultLogger, Config
        print("  ✓ target_tracker module imported successfully")
        
        # Test basic instantiation
        try:
            tracker = TargetTracker()
            print("  ✓ TargetTracker instance created")
            
            # Test config values
            print(f"  ✓ Frame size: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
            print(f"  ✓ Min area: {Config.MIN_AREA} pixels")
            print(f"  ✓ HSV range 1: {Config.HSV_LOWER1} - {Config.HSV_UPPER1}")
            print(f"  ✓ HSV range 2: {Config.HSV_LOWER2} - {Config.HSV_UPPER2}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ TargetTracker instantiation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"  ✗ target_tracker import failed: {e}")
        print("    Make sure target_tracker.py is in the same directory")
        return False

def test_opencv_camera():
    """Test if OpenCV can access a camera"""
    print("\nTesting OpenCV camera access...")
    
    try:
        import cv2
        
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("  ✓ OpenCV can access camera (index 0)")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"  ✓ Frame captured successfully: {width}x{height}")
            else:
                print("  ⚠ Camera opened but failed to read frame")
            
            cap.release()
            return True
        else:
            print("  ✗ OpenCV cannot access camera (index 0)")
            print("    This might be normal if no USB camera is connected")
            return False
            
    except Exception as e:
        print(f"  ✗ Camera test failed: {e}")
        return False

def test_file_permissions():
    """Test if tracker scripts exist and are executable"""
    print("\nTesting tracker script files...")
    
    scripts = [
        'main_pi.py',
        'main_pi_zero.py', 
        'main_pi_headless.py',
        'main_macos.py'
    ]
    
    import os
    
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script} exists")
            if os.access(script, os.X_OK):
                print(f"  ✓ {script} is executable")
            else:
                print(f"  ⚠ {script} is not executable (run: chmod +x {script})")
        else:
            print(f"  ✗ {script} not found")

def main():
    """Run all tests"""
    print("=" * 60)
    print("RASPBERRY PI CAMERA INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    
    # Run all tests
    import_issues = test_imports()
    picamera_ok = test_picamera2()
    tracker_ok = test_target_tracker()
    camera_ok = test_opencv_camera()
    test_file_permissions()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if import_issues:
        print("❌ IMPORT ISSUES FOUND:")
        for module, package in import_issues:
            print(f"   Missing: {module} (install: pip install {package})")
        print()
    
    if tracker_ok:
        print("✅ Target tracker module: OK")
    else:
        print("❌ Target tracker module: FAILED")
    
    if picamera_ok:
        print("✅ Pi camera integration: READY")
        print("   You can run: python3 main_pi.py")
    else:
        print("⚠️  Pi camera integration: NOT AVAILABLE")
        print("   This is normal on non-Pi systems")
        print("   On Pi: check camera connection and enable with raspi-config")
    
    if camera_ok:
        print("✅ USB/Webcam access: OK")
        print("   You can run: python3 target_tracker.py --camera")
    else:
        print("ℹ️  USB/Webcam access: NO CAMERA DETECTED")
        print("   This is normal if no USB camera is connected")
    
    print()
    
    if import_issues:
        print("❌ SETUP INCOMPLETE - Install missing dependencies")
        return 1
    elif not tracker_ok:
        print("❌ SETUP INCOMPLETE - Check target_tracker.py")
        return 1
    else:
        print("✅ SETUP COMPLETE - Ready for target tracking!")
        
        print("\nNext steps:")
        if picamera_ok:
            print("  • On Raspberry Pi: python3 main_pi.py")
        if camera_ok:
            print("  • With USB camera: python3 target_tracker.py --camera")
            print("  • With macOS webcam: python3 main_macos.py") 
        print("  • With demo video: python3 target_tracker.py demo.mp4")
        print("  • Read setup guide: cat PI_CAMERA_SETUP.md")
        
        return 0

if __name__ == "__main__":
    sys.exit(main()) 