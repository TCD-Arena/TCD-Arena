#!/bin/bash

# This script generates all observational noise datasets.
# For datasets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=1000 seed=$seed n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1"
cmd3=" generator.inno_n.non_additive_noise_proba=0.475,0.60625,0.7375,0.86875,1"
# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd2="$cmd2 output_dir=$output_dir"
fi

specs=(
 "inno_common_2_3 n_vars=2 generator.lagged.max_lags=3 generator.inno_n.common=True"
)

for spec in "${specs[@]}"; do
    echo $cmd$spec$cmd2$cmd3
    eval $cmd$spec$cmd2$cmd3
done

wait
echo "Done"
