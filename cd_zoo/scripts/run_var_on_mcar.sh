#!/bin/bash

# Simplified: Run only the 'var' method for all folders in sample_datasets/mcar


cmd="python cd_zoo/benchmark.py -m method=var data_base_path=sample_datasets/mcar/ method.base_on=coefficients,p_values"
cmd2="ds_name="
cmd3="which_dataset='range(0,40)'"

for data_path in sample_datasets/mcar/*/; do
    # Skip if no folders match the pattern
    if [[ ! -d "$data_path" ]]; then
        continue
    fi
    folder_name=$(basename "$data_path")
    echo "Running VAR method for $data_path"
    # Set max_lag based on folder name if needed, else use default
    if [[ "$folder_name" == *"big"* ]]; then
        cmd4=" method.max_lag=4"
    elif [[ "$folder_name" == *"small"* ]]; then
        cmd4=" method.max_lag=3"
    else
        echo "Error: Dataset folder $data_path does not specify 'big' or 'small' in its name."
        exit 1
    fi
    echo "$cmd $cmd2$folder_name $cmd3$cmd4"
    eval "$cmd $cmd2$folder_name $cmd3$cmd4"
    
done

echo "Done"