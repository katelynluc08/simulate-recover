import sys
import os
import unittest
import numpy as np

# Add the src directory to the Python path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions from your implementation
# Assuming your main code is in a file called ez_diffusion_model.py in the src directory
from ez_diffusion_model import random_parameters, forward_eq, inverse_eq, simulate_R_obs, simulate_M_obs, simulate_V_obs

class TestEZDiffusionModel(unittest.TestCase):
    
    def test_random_parameters(self):
        """Test that random_parameters generates values within expected ranges"""
        a, v, t = random_parameters()
        self.assertTrue(0.5 <= a <= 2, f"Boundary separation {a} outside range [0.5, 2]")
        self.assertTrue(0.5 <= v <= 2, f"Drift rate {v} outside range [0.5, 2]")
        self.assertTrue(0.1 <= t <= 0.5, f"Non-decision time {t} outside range [0.1, 0.5]")
    
    def test_forward_eq(self):
        """Test forward equation with known values"""
        # Test with a simple case
        a, v, t = 1.0, 1.0, 0.2
        results = forward_eq(a, v, t)
        
        # Expected values calculated by hand
        y = np.exp(-a * v)
        expected_R = 1 / (y + 1)
        expected_M = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        expected_V = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((y + 1) ** 2))
        
        self.assertAlmostEqual(results["R_pred"], expected_R, places=5)
        self.assertAlmostEqual(results["M_pred"], expected_M, places=5)
        self.assertAlmostEqual(results["V_pred"], expected_V, places=5)
    
    def test_inverse_recovery(self):
        """Test that parameters can be recovered through the inverse equations"""
        # Generate some parameters
        a_orig, v_orig, t_orig = 1.0, 1.0, 0.2
        
        # Get predicted values through forward equation
        pred = forward_eq(a_orig, v_orig, t_orig)
        
        # Use a large sample size to minimize noise
        N = 10000
        
        # Simulate observed values with minimal noise
        R_obs = pred["R_pred"]  # Use exact value to focus on equation correctness
        M_obs = pred["M_pred"]
        V_obs = pred["V_pred"]
        
        # Recover parameters
        v_est, a_est, t_est = inverse_eq(R_obs, V_obs, M_obs)
        
        # Check if recovered parameters are close to original
        # Allow some numerical error but should be close with exact inputs
        self.assertAlmostEqual(a_orig, a_est, places=2)
        self.assertAlmostEqual(v_orig, v_est, places=2)
        self.assertAlmostEqual(t_orig, t_est, places=2)
    
    def test_simulation_functions(self):
        """Test that simulation functions behave as expected"""
        # Test with simple values
        R_pred = 0.75
        M_pred = 0.5
        V_pred = 0.1
        N = 1000
        
        # Simulate observations
        R_obs = simulate_R_obs(R_pred, N)
        M_obs = simulate_M_obs(M_pred, V_pred, N)
        V_obs = simulate_V_obs(V_pred, N)
        
        # Check that results are reasonable (close to expected values)
        self.assertTrue(0 <= R_obs <= 1, f"R_obs={R_obs} outside valid range [0,1]")
        self.assertTrue(abs(R_obs - R_pred) < 0.1, f"R_obs={R_obs} far from R_pred={R_pred}")
        self.assertTrue(abs(M_obs - M_pred) < 0.1, f"M_obs={M_obs} far from M_pred={M_pred}")
        self.assertTrue(V_obs > 0, f"V_obs={V_obs} should be positive")
        self.assertTrue(abs(V_obs - V_pred) < 0.1, f"V_obs={V_obs} far from V_pred={V_pred}")

if __name__ == "__main__":
    unittest.main()
