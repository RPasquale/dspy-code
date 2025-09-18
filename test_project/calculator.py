"""
Simple Calculator Module

A basic calculator with arithmetic operations.
"""

class Calculator:
    """A simple calculator class with basic arithmetic operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    
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

    def get_history(self):
        """Get calculation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()
