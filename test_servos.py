"""
Simple servo test script for the Pi Zero target tracking system.
Tests servo movement and positioning to verify hardware setup.
"""

import time
import argparse

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
    GPIO_AVAILABLE = False

class ServoTester:
    """Simple servo tester for hardware verification."""
    
    def __init__(self, pins=[18, 19, 20, 21], pwm_frequency=50):
        self.pins = pins
        self.pwm_frequency = pwm_frequency
        self.pwm_objects = []
        self.initialized = False
        
        self.initialize_servos()
    
    def initialize_servos(self):
        """Initialize GPIO and PWM for servo control."""
        if not GPIO_AVAILABLE:
            print("Simulation mode - servo movements will be printed")
            self.initialized = True
            return
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            for pin in self.pins:
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, self.pwm_frequency)
                pwm.start(7.5)  # Start at 90 degrees
                self.pwm_objects.append(pwm)
            
            print(f"Servos initialized on pins {self.pins}")
            self.initialized = True
            
        except Exception as e:
            print(f"Failed to initialize servos: {e}")
            self.initialized = False
    
    def angle_to_duty_cycle(self, angle):
        """Convert servo angle to PWM duty cycle."""
        min_duty = 2.5  # 0.5ms pulse width
        max_duty = 12.5  # 2.5ms pulse width
        duty_cycle = min_duty + (angle / 180.0) * (max_duty - min_duty)
        return duty_cycle
    
    def set_angle(self, servo_index, angle):
        """Set a specific servo to a specific angle."""
        if not self.initialized:
            return
        
        angle = max(0, min(180, angle))  # Clamp to valid range
        
        if GPIO_AVAILABLE and servo_index < len(self.pwm_objects):
            duty_cycle = self.angle_to_duty_cycle(angle)
            self.pwm_objects[servo_index].ChangeDutyCycle(duty_cycle)
            print(f"Servo {servo_index} (pin {self.pins[servo_index]}): {angle}°")
        else:
            print(f"[SIM] Servo {servo_index} (pin {self.pins[servo_index]}): {angle}°")
    
    def set_all_angles(self, angles):
        """Set all servos to specified angles."""
        for i, angle in enumerate(angles):
            if i < len(self.pins):
                self.set_angle(i, angle)
    
    def test_center_position(self):
        """Test center position (90 degrees)."""
        print("\n=== Testing Center Position (90°) ===")
        self.set_all_angles([90, 90, 90, 90])
        time.sleep(2)
    
    def test_corner_positions(self):
        """Test corner positions."""
        print("\n=== Testing Corner Positions ===")
        
        # Top-left corner (target at top-left)
        print("Target at top-left corner:")
        self.set_all_angles([135, 90, 90, 45])  # TL max, others compensate
        time.sleep(2)
        
        # Top-right corner (target at top-right)
        print("Target at top-right corner:")
        self.set_all_angles([90, 135, 45, 90])  # TR max, others compensate
        time.sleep(2)
        
        # Bottom-left corner (target at bottom-left)
        print("Target at bottom-left corner:")
        self.set_all_angles([90, 45, 135, 90])  # BL max, others compensate
        time.sleep(2)
        
        # Bottom-right corner (target at bottom-right)
        print("Target at bottom-right corner:")
        self.set_all_angles([45, 90, 90, 135])  # BR max, others compensate
        time.sleep(2)
    
    def test_edge_positions(self):
        """Test edge positions."""
        print("\n=== Testing Edge Positions ===")
        
        # Top edge
        print("Target at top edge:")
        self.set_all_angles([112.5, 112.5, 67.5, 67.5])
        time.sleep(2)
        
        # Right edge
        print("Target at right edge:")
        self.set_all_angles([67.5, 112.5, 67.5, 112.5])
        time.sleep(2)
        
        # Bottom edge
        print("Target at bottom edge:")
        self.set_all_angles([67.5, 67.5, 112.5, 112.5])
        time.sleep(2)
        
        # Left edge
        print("Target at left edge:")
        self.set_all_angles([112.5, 67.5, 112.5, 67.5])
        time.sleep(2)
    
    def test_sweep(self):
        """Test smooth sweep movement."""
        print("\n=== Testing Smooth Sweep ===")
        
        for angle in range(45, 136, 5):
            print(f"All servos to {angle}°")
            self.set_all_angles([angle, angle, angle, angle])
            time.sleep(0.2)
        
        for angle in range(135, 44, -5):
            print(f"All servos to {angle}°")
            self.set_all_angles([angle, angle, angle, angle])
            time.sleep(0.2)
    
    def cleanup(self):
        """Clean up GPIO resources."""
        if GPIO_AVAILABLE and self.initialized:
            # Stop PWM objects first
            for pwm in self.pwm_objects:
                try:
                    pwm.stop()
                except:
                    pass  # Ignore errors if already stopped
            
            # Clear the PWM objects list to prevent destructor issues
            self.pwm_objects.clear()
            
            # Clean up GPIO
            GPIO.cleanup()
            print("GPIO cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='Test servo functionality for Pi Zero target tracker')
    parser.add_argument('--pins', nargs=4, type=int, default=[18, 19, 20, 21],
                        help='GPIO pins for servos [TL, TR, BL, BR] (default: 18 19 20 21)')
    parser.add_argument('--test', choices=['center', 'corners', 'edges', 'sweep', 'all'], 
                        default='all', help='Test to run (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SERVO TEST FOR PI ZERO TARGET TRACKER")
    print("=" * 50)
    print(f"Testing servos on pins: {args.pins}")
    print("Servo layout: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]")
    print("Angle range: 45-135° (center at 90°)")
    print()
    
    tester = ServoTester(pins=args.pins)
    
    if not tester.initialized:
        print("Failed to initialize servos. Check connections and permissions.")
        return
    
    try:
        if args.test == 'center' or args.test == 'all':
            tester.test_center_position()
        
        if args.test == 'corners' or args.test == 'all':
            tester.test_corner_positions()
        
        if args.test == 'edges' or args.test == 'all':
            tester.test_edge_positions()
        
        if args.test == 'sweep' or args.test == 'all':
            tester.test_sweep()
        
        print("\n=== Returning to Center Position ===")
        tester.test_center_position()
        
        print("\nServo test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main() 