#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate animusicator

# Install required packages
conda install -y numpy scipy pillow pyside6 pyopengl

# Set up Python path for the project
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Environment setup complete!"
echo "Use 'conda activate animusicator' to activate the environment"
echo "Run your app with: python -m musicviz.main" 
