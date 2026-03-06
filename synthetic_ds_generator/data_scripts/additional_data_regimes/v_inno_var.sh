#!/bin/bash

# Super ugly... Fix this in the future.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=1000 n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1 seed=$seed"
cmd3=" generator.inno_n.non_equal_variance_range='[0.55,0.75,1.25,1.45],[0.4125,0.6125,1.3875,1.5875],[0.275,0.475,1.525,1.725],[0.1375,0.3375,1.6625,1.8625],[0.,0.2,1.8,2]'"

# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd2="$cmd2 output_dir=$output_dir"
fi

specs=(
 "inno_var_7_2 n_vars=7 generator.lagged.max_lags=2"
 "inno_var_3_2 n_vars=3 generator.lagged.max_lags=2"
 "inno_var_3_4 n_vars=3 generator.lagged.max_lags=4"
)

for spec in "${specs[@]}"; do
    echo $cmd$spec$cmd2$cmd3
    eval $cmd$spec$cmd2$cmd3
done

wait
echo "Done"
