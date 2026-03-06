#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=250,1000 n_samples=100 seed=$seed generator.lagged.link_proba=0.075,0.15  generator.instant.link_proba=0.,0.1"
cmd4=" generator.obs_n.snr=1.1,0.8375,0.575,0.3125,0.05"

# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd4="$cmd4 output_dir=$output_dir"
fi

### Increased observation noise

specs=(
 "obs_add_big generator.obs_n.additive=True"
 "obs_add_small generator.obs_n.additive=True"
 "obs_mul_big generator.obs_n.multiplicative=True"
 "obs_mul_small generator.obs_n.multiplicative=True"
 "obs_time_big generator.obs_n.time_dependent=True"
 "obs_time_small generator.obs_n.time_dependent=True"
 "obs_auto_big generator.obs_n.autoregressive=True"
 "obs_auto_small generator.obs_n.autoregressive=True"
 "obs_com_big generator.obs_n.common=True"
 "obs_com_small generator.obs_n.common=True"
 "obs_shock_big generator.obs_n.shock=True"
 "obs_shock_small generator.obs_n.shock=True"
 "obs_real_big generator.obs_n.semi_synthetic=True"
 "obs_real_small generator.obs_n.semi_synthetic=True"
)


for spec in "${specs[@]}"; do
    if [[ "$spec" == *"big"* ]]; then
        cmd3=" n_vars=7 generator.lagged.max_lags=4" 
    fi 
    if [[ "$spec" == *"small"* ]]; then
        cmd3=" n_vars=5 generator.lagged.max_lags=3" 
    fi

    echo  $cmd$spec$cmd2$cmd3$cmd4
    eval $cmd$spec$cmd2$cmd3$cmd4 

done

wait
echo "Done"
