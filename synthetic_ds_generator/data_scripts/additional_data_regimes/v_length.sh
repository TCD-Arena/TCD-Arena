#!/bin/bash


# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=76,58,41,23,6 n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1 seed=$seed"

# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd2="$cmd2 output_dir=$output_dir"
fi

specs=(
 "length_7_2 n_vars=7 generator.lagged.max_lags=2"
 "length_3_2 n_vars=3 generator.lagged.max_lags=2"
 "length_3_4 n_vars=3 generator.lagged.max_lags=4"
)

for spec in "${specs[@]}"; do
    echo $cmd$spec$cmd2
    eval $cmd$spec$cmd2
done

wait
echo "Done"
