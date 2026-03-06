#!/bin/bash

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m name="
cmd2=" generator.lagged.link_proba=0 n_samples=100 generator.instant.link_proba=0.,0.1 generator.time_series_n=250,1000  seed=$seed"


# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi


# Options for the severity
cmd3=' link_mask_path="tools_and_examples/masks/masks_5_4_faith_lagged_4","tools_and_examples/masks/masks_5_4_faith_lagged_3","tools_and_examples/masks/masks_5_4_faith_lagged_2","tools_and_examples/masks/masks_5_4_faith_lagged_1","tools_and_examples/masks/masks_5_4_faith_lagged_0"'
cmd4=' link_mask_path="tools_and_examples/masks/masks_5_2_faith_lagged_4","tools_and_examples/masks/masks_5_2_faith_lagged_3","tools_and_examples/masks/masks_5_2_faith_lagged_2","tools_and_examples/masks/masks_5_2_faith_lagged_1","tools_and_examples/masks/masks_5_2_faith_lagged_0"'
cmd5=' link_mask_path="tools_and_examples/masks/masks_7_4_faith_lagged_4","tools_and_examples/masks/masks_7_4_faith_lagged_3","tools_and_examples/masks/masks_7_4_faith_lagged_2","tools_and_examples/masks/masks_7_4_faith_lagged_1","tools_and_examples/masks/masks_7_4_faith_lagged_0"'
cmd6=' link_mask_path="tools_and_examples/masks/masks_7_6_faith_lagged_4","tools_and_examples/masks/masks_7_6_faith_lagged_3","tools_and_examples/masks/masks_7_6_faith_lagged_2","tools_and_examples/masks/masks_7_6_faith_lagged_1","tools_and_examples/masks/masks_7_6_faith_lagged_0"'


### Increased observation noise

specs=(
    "faith_lagged_small$cmd3"
    "faith_lagged_small$cmd4"
    "faith_lagged_big$cmd5"
    "faith_lagged_big$cmd6"
)


for spec in "${specs[@]}"; do
    if [[ "$spec" == *"big"* ]]; then
        cmd9=" n_vars=7 generator.lagged.max_lags=4" 
    fi 
    if [[ "$spec" == *"small"* ]]; then
        cmd9=" n_vars=5 generator.lagged.max_lags=3" 
    fi

    echo  $cmd$spec$cmd9$cmd2$output_dir_param
    eval $cmd$spec$cmd9$cmd2$output_dir_param


done

wait
echo "Done"
