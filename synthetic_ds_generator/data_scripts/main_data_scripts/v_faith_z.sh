#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


BASE_CMD="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=250,1000 seed=$seed n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1"


# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi


# Five single-level param ranges
levels=(
    '"[0.12,0.24]"'
    '"[0.0975,0.1925]"'
    '"[0.075,0.145]"'
    '"[0.0525,0.0975]"'
    '"[0.03,0.05]"'
)

# Build five command variants (lagged and instant)
cmds_levels=()
for lvl in "${levels[@]}"; do
    cmds_levels+=(" generator.lagged.param_range=$lvl generator.instant.param_range=$lvl")
done



specs=(
    # Multiplicative noise
    "faith_z_small "
    "faith_z_big "
)

for spec in "${specs[@]}"; do
    # Set variable count and max lags based on dataset size
    if [[ "$spec" == *"big"* ]]; then
        VAR_ARGS=" n_vars=7 generator.lagged.max_lags=4"
    elif [[ "$spec" == *"small"* ]]; then
        VAR_ARGS=" n_vars=5 generator.lagged.max_lags=3"
    fi
    for cmd3 in "${cmds_levels[@]}"; do
        # Build and print the full command for clarity
        FULL_CMD="$BASE_CMD$spec$cmd2$cmd3$VAR_ARGS$output_dir_param"
        echo "$FULL_CMD"
        eval "$FULL_CMD" 
    done
done

wait
echo "Done"
