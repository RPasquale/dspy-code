#!/usr/bin/env python3
"""
Test script to implement and test the power function for the calculator.
This demonstrates a simpler approach than using the agent's edit command.
"""

import sys
import os
sys.path.append('/Users/robbiepasquale/dspy_stuff/test_project')

def add_power_function_to_calculator():
    """Add power function to calculator.py"""
    
    # Read the current calculator.py
    calc_file = '/Users/robbiepasquale/dspy_stuff/test_project/calculator.py'
    
    with open(calc_file, 'r') as f:
        content = f.read()
    
    # Add the power function before the get_history method
    power_function = '''    
    def power(self, base, exponent):
        """Calculate base raised to the power of exponent."""
        # Handle edge cases
        if exponent == 0:
            result = 1
        elif exponent < 0:
            # Handle negative exponents
            if base == 0:
                raise ValueError("Cannot raise 0 to a negative power")
            result = 1 / (base ** abs(exponent))
        else:
            result = base ** exponent
        
        # Add to history
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result
'''
    
    # Insert the power function before get_history method
    insertion_point = content.find('    def get_history(self):')
    if insertion_point == -1:
        print("Error: Could not find insertion point")
        return False
    
    new_content = content[:insertion_point] + power_function + '\n' + content[insertion_point:]
    
    # Write back to file
    with open(calc_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Added power function to calculator.py")
    return True

def add_power_tests():
    """Add test cases for the power function"""
    
    test_file = '/Users/robbiepasquale/dspy_stuff/test_project/test_calculator.py'
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Add power test methods
    power_tests = '''
    def test_power(self):
        """Test power functionality."""
        assert self.calc.power(2, 3) == 8
        assert self.calc.power(5, 2) == 25
        assert self.calc.power(10, 0) == 1
        assert self.calc.power(-2, 3) == -8
    
    def test_power_negative_exponent(self):
        """Test power with negative exponents."""
        assert self.calc.power(2, -2) == 0.25
        assert self.calc.power(4, -1) == 0.25
    
    def test_power_zero_base(self):
        """Test power with zero base and negative exponent."""
        with pytest.raises(ValueError, match="Cannot raise 0 to a negative power"):
            self.calc.power(0, -1)
    
    def test_power_history(self):
        """Test power operation is added to history."""
        self.calc.power(3, 2)
        history = self.calc.get_history()
        assert "3 ^ 2 = 9" in history
'''
    
    # Insert before the last method
    insertion_point = content.rfind('    def test_clear_history(self):')
    if insertion_point == -1:
        print("Error: Could not find test insertion point")
        return False
    
    new_content = content[:insertion_point] + power_tests + '\n' + content[insertion_point:]
    
    # Write back to file
    with open(test_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Added power function tests to test_calculator.py")
    return True

def run_tests():
    """Run the tests to verify the implementation"""
    import subprocess
    
    os.chdir('/Users/robbiepasquale/dspy_stuff/test_project')
    
    try:
        result = subprocess.run(['uv', 'run', 'pytest', 'test_calculator.py', '-v'], 
                              capture_output=True, text=True, timeout=30)
        
        print("🧪 Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Implementing Power Function for Calculator")
    print("=" * 50)
    
    # Step 1: Add power function
    if not add_power_function_to_calculator():
        sys.exit(1)
    
    # Step 2: Add tests
    if not add_power_tests():
        sys.exit(1)
    
    # Step 3: Run tests
    print("\n📋 Running tests...")
    if run_tests():
        print("\n🎉 SUCCESS: Power function implemented and all tests pass!")
    else:
        print("\n❌ FAILED: Some tests failed")
        sys.exit(1)
