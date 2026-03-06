#!/bin/bash

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.lagged.link_proba=0.075,0.15 n_samples=100 generator.instant.link_proba=0. generator.time_series_n=250,1000 seed=$seed"

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi


### Increased observation noise



specs=()

for size in small big; do
    if [[ "$size" == "small" ]]; then
        n=5
    else
        n=7
    fi

    mask_paths=()
    for i in 4 3 2 1 0; do
        mask_paths+=("tools_and_examples/masks/masks_${n}_faith_instant_${i}")
    done
    mask_paths_str=$(IFS=,; echo "${mask_paths[*]}")

    specs+=("faith_inst_${size} instant_link_mask_path=${mask_paths_str}")
done


for spec in "${specs[@]}"; do
    if [[ "$spec" == *"big"* ]]; then
        cmd9=" n_vars=7 generator.lagged.max_lags=4" 
    fi 
    if [[ "$spec" == *"small"* ]]; then

        cmd9=" n_vars=5 generator.lagged.max_lags=3" 
    fi

    echo  $cmd$spec$cmd2$cmd9$output_dir_param
    eval $cmd$spec$cmd2$cmd9$output_dir_param


done

wait
echo "Done"
