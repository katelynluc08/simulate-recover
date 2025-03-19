#!/bin/bash

# This script runs the test suite for the EZ diffusion model implementation

echo "Running test suite for EZ Diffusion Model..."

# Navigate to the directory containing the tests
cd "$(dirname "$0")"

# Run the Python test script
# Assuming you have a test script named test_ez_diffusion.py in the test directory
python test_ez_diffusion.py

echo "Test suite completed."
