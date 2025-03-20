# This script runs the complete 3000-iteration simulate-and-recovery exercise
# for the EZ diffusion model consistency test

echo "Starting EZ Diffusion Model Consistency Test (3000 iterations)..."

# Navigate to the directory containing the script
# Assuming this script is in the src directory
cd "$(dirname "$0")"

# Run the Python script for the simulate-and-recover exercise
python ez_diffusion_model.py

echo "Consistency test completed."
