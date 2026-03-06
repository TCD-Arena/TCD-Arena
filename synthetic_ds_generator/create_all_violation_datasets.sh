#!/bin/bash

# This script runs all violation generation scripts with the provided arguments
# Usage: ./create_all_violations.sh [--regime REGIME] [seed] [output_dir]
# Example: ./create_all_violations.sh 4242 /path/to/output
# Example: ./create_all_violations.sh --regime additional_data_regimes 4242 /path/to/output

# Default regime
REGIME="main_data_scripts"

# Parse optional --regime flag
if [ "$1" == "--regime" ]; then
    REGIME="$2"
    shift 2
fi

SCRIPT_DIR="$(dirname "$0")/data_scripts/$REGIME"
# Pass remaining arguments to each script
ARGS="$@"

echo "Using regime: $REGIME"
echo "Running all violation scripts with arguments: $ARGS"


for script in "$SCRIPT_DIR"/*.sh; do
    if [ -x "$script" ]; then
        echo "Running $script $ARGS"
        "$script" $ARGS
    else
        echo "Skipping $script (not executable)"
    fi
done

echo "All violation scripts completed"