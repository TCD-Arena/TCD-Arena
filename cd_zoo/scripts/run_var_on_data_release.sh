#!/bin/bash

# Usage: ./run_var_on_data_release.sh [filter_string] [limit_n]
#   filter_string: only process folders containing this string (optional)
#   limit_n: only process the first n matching folders (optional)

limit_n="$1"
filter_string="$2"

cmd="python cd_zoo/benchmark.py -m method=var data_base_path=data_release/ method.base_on=coefficients,p_values"
cmd2="ds_name="
cmd3="which_dataset='range(0,40)'"

count=0
for data_path in data_release/*/; do
    # Skip if no folders match the pattern
    if [[ ! -d "$data_path" ]]; then
        continue
    fi
    folder_name=$(basename "$data_path")

    # If filter_string is set, skip folders that don't match
    if [[ -n "$filter_string" && "$folder_name" != *"$filter_string"* ]]; then
        continue
    fi

    # If limit_n is set, stop after n matches
    if [[ -n "$limit_n" && $count -ge $limit_n ]]; then
        break
    fi

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
    count=$((count+1))
done

echo "Done"