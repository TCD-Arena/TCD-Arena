#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=250,1000 n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1 seed=$seed"
cmd3=" nonlinear_proba=1"
cmd4=' nl_mode="power_set"'
cmd5=' power_dist="[0.5,1,1,2]","[0.25,0.5,2,4]","[0.125,0.25,4,8]","[0.0833,0.125,8,12]","[0.05,0.0833,12,20]"'

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi


specs=(
    "nl_mono_small"
    "nl_mono_big"
)


for spec in "${specs[@]}"; do
    if [[ "$spec" == *"big"* ]]; then
        cmd6=" n_vars=7 generator.lagged.max_lags=4" 
    fi 
    if [[ "$spec" == *"small"* ]]; then
        cmd6=" n_vars=5 generator.lagged.max_lags=3" 
    fi


    echo $cmd$spec$cmd2$cmd3$cmd4$cmd5$cmd6$output_dir_param
    eval $cmd$spec$cmd2$cmd3$cmd4$cmd5$cmd6$output_dir_param 

done

wait
echo "Done"
