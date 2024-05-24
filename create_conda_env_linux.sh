#!/bin/bash

# Name of the Conda environment
ENV_NAME="ocdr"

# Path to the requirements.txt file
REQUIREMENTS_FILE="requirements.txt"

# Function to check if the Conda environment exists
env_exists() {
    conda info --envs | grep "^$ENV_NAME" > /dev/null
}

# Activate Conda environment
source activate "$ENV_NAME" 2> /dev/null

if env_exists; then
    echo "Environment '$ENV_NAME' exists, updating..."
    # If the environment exists, activate and install packages
    source activate "$ENV_NAME"
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Environment '$ENV_NAME' does not exist, creating..."
    # Create the environment with Python and install packages
    conda create --name "$ENV_NAME" python=3.10 --yes
    conda activate "$ENV_NAME"
    pip install -r "$REQUIREMENTS_FILE"
fi

echo "Operation completed."
