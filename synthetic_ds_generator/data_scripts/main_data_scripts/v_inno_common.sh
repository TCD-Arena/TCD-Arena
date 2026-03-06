#!/bin/bash

# This script generates all observational noise datasets.
# For datasets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


# Base command parts
BASE_CMD="python generate_dataset.py -m  name="
COMMON_ARGS=" generator.time_series_n=250,1000 n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1 seed=$seed"
NOISE_PROB_ARGS=" generator.inno_n.non_additive_noise_proba=0.475,0.60625,0.7375,0.86875,1"

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi

# Dataset specifications
specs=(
    # CausalRivers based semi synthetic noise
    "inno_common_big generator.inno_n.common=True"
    "inno_common_small generator.inno_n.common=True"
)

for spec in "${specs[@]}"; do
    # Set variable count and max lags based on dataset size
    if [[ "$spec" == *"big"* ]]; then
        VAR_ARGS=" n_vars=7 generator.lagged.max_lags=4"
    elif [[ "$spec" == *"small"* ]]; then
        VAR_ARGS=" n_vars=5 generator.lagged.max_lags=3"
    else
        VAR_ARGS=""
    fi

    # Build and print the full command for clarity
    FULL_CMD="$BASE_CMD$spec$COMMON_ARGS$VAR_ARGS$NOISE_PROB_ARGS$output_dir_param"
    echo "$FULL_CMD"
    eval "$FULL_CMD" 
done

wait
echo "Done"
