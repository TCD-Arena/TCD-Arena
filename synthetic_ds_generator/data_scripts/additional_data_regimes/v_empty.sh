#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m"
cmd2=" generator.lagged.link_proba=0.075,0.15 n_samples=100 generator.instant.link_proba=0.,0.1 generator.drop_struc_for_window=True seed=$seed"
cmd3=' generator.time_series_n=1000 generator.change_points="[160,400,600,840]","[124,424,576,876]","[92,444,556,908]","[56,468,532,944]","[24,488,512,976]"' 

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi

### Increased observation noise

specs=(
    "$cmd name=empty_7_2$cmd2 n_vars=7 generator.lagged.max_lags=2 nonstationary_change=0 generator.nonstationary=True$cmd3$output_dir_param"
    "$cmd name=empty_3_2$cmd2 n_vars=3 generator.lagged.max_lags=2 nonstationary_change=0 generator.nonstationary=True$cmd3$output_dir_param"
    "$cmd name=empty_3_4$cmd2 n_vars=3 generator.lagged.max_lags=4 nonstationary_change=0 generator.nonstationary=True$cmd3$output_dir_param"
)

echo $specs


for spec in "${specs[@]}"; do
    eval $spec 

done

wait
echo "Done"
