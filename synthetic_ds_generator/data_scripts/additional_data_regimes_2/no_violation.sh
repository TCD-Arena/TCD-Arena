#!/bin/bash

# This generates a clean synthetic dataset with no violations for all data regimes tested.
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}

cmd="python generate_dataset.py -m"
cmd2=" generator.lagged.link_proba=0.075,0.15 n_samples=100 generator.instant.link_proba=0.,0.1 generator.time_series_n=1000 seed=$seed"

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir/"
fi 

specs=(
    "$cmd name=no_violation_2_3$cmd2 n_vars=2 generator.lagged.max_lags=3$output_dir_param"

)

for spec in "${specs[@]}"; do
    echo "Running: $spec"
    eval $spec 
done

wait
echo "Done"
