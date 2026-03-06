#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=1000 n_samples=100 seed=$seed generator.lagged.link_proba=0.075,0.15  generator.instant.link_proba=0.,0.1"
cmd3=" generator.obs_n.snr=1.1,0.8375,0.575,0.3125,0.05"

# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd2="$cmd2 output_dir=$output_dir"
fi

specs=(
 "obs_add_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.additive=True"
 "obs_mul_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.multiplicative=True"
 "obs_time_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.time_dependent=True"
 "obs_auto_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.autoregressive=True"
 "obs_com_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.common=True"
 "obs_shock_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.shock=True"
 "obs_real_2_3 n_vars=2 generator.lagged.max_lags=3 generator.obs_n.semi_synthetic=True"
)

for spec in "${specs[@]}"; do
    echo $cmd$spec$cmd2$cmd3
    eval $cmd$spec$cmd2$cmd3
done 

done

wait
echo "Done"
