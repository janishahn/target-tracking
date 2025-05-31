"""
Test script to verify target tracker installation and dependencies.
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_camera_access():
    """Test if camera can be accessed"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✓ Camera access successful")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame capture successful: {frame.shape}")
            else:
                print("⚠ Camera opened but frame capture failed")
            cap.release()
            return True
        else:
            print("⚠ Camera could not be opened (this is normal if no camera is connected)")
            return True  # Not a failure for systems without cameras
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_target_tracker():
    """Test if target_tracker.py can be imported"""
    print("\nTesting target tracker module...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Try to import the main components
        from target_tracker import Config, TargetTracker, circularity_of
        print("✓ Target tracker components imported successfully")
        
        # Test basic functionality
        tracker = TargetTracker()
        print("✓ TargetTracker instance created successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Target tracker import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Target tracker test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Target Tracker Installation Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_camera_access():
        tests_passed += 1
    
    if test_target_tracker():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Target tracker is ready to use.")
        print("\nNext steps:")
        print("1. Create or obtain a demo video with a red circular object")
        print("2. Run: python target_tracker.py --source demo.mp4 --debug")
        print("3. Or test with camera: python target_tracker.py --camera --debug")
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 