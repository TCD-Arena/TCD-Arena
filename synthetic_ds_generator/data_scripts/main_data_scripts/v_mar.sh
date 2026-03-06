#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=250,1000 n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0,0.1 seed=$seed"
cmd3=" generator.interpolate=0.2,0.3375,0.475,0.6125,0.75 generator.missingness_type=MAR"

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi

specs=(
    "mar_small"
    "mar_big"
)


for spec in "${specs[@]}"; do
    if [[ "$spec" == *"big"* ]]; then
        cmd8=" n_vars=7 generator.lagged.max_lags=4" 
    fi 
    if [[ "$spec" == *"small"* ]]; then
        cmd8=" n_vars=5 generator.lagged.max_lags=3" 
    fi


    echo $cmd$spec$cmd2$cmd3$cmd8$output_dir_param
    eval $cmd$spec$cmd2$cmd3$cmd8$output_dir_param 

done

wait
echo "Done"
