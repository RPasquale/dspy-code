"""
Tests for the Calculator module.
"""

import pytest
from calculator import Calculator


class TestCalculator:
    """Test cases for Calculator class."""
    
    def setup_method(self):
        """Set up a fresh calculator instance for each test."""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition functionality."""
        assert self.calc.add(2, 3) == 5
        assert self.calc.add(-1, 1) == 0
        assert self.calc.add(0, 0) == 0
    
    def test_subtract(self):
        """Test subtraction functionality."""
        assert self.calc.subtract(5, 3) == 2
        assert self.calc.subtract(1, 1) == 0
        assert self.calc.subtract(0, 5) == -5
    
    def test_multiply(self):
        """Test multiplication functionality."""
        assert self.calc.multiply(2, 3) == 6
        assert self.calc.multiply(-2, 3) == -6
        assert self.calc.multiply(0, 5) == 0
    
    def test_divide(self):
        """Test division functionality."""
        assert self.calc.divide(6, 2) == 3
        assert self.calc.divide(5, 2) == 2.5
        assert self.calc.divide(-6, 2) == -3
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(5, 0)
    
    def test_history(self):
        """Test history tracking."""
        self.calc.add(1, 2)
        self.calc.multiply(3, 4)
        history = self.calc.get_history()
        assert len(history) == 2
        assert "1 + 2 = 3" in history
        assert "3 * 4 = 12" in history
    

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

    def test_clear_history(self):
        """Test clearing history."""
        self.calc.add(1, 2)
        self.calc.clear_history()
        assert len(self.calc.get_history()) == 0
