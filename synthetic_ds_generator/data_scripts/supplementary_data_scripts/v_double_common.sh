#!/bin/bash

# This generates a multi violationd dataset.
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=250,1000 n_samples=100 seed=$seed generator.lagged.link_proba=0.075,0.15  generator.instant.link_proba=0.,0.1 generator.inno_n.common=True generator.obs_n.common=True"
cmd4=" generator.obs_n.snr=1.1 generator.inno_n.non_additive_noise_proba=0.475"
cmd4=" generator.obs_n.snr=0.8375 generator.inno_n.non_additive_noise_proba=0.60625"
cmd4=" generator.obs_n.snr=0.575 generator.inno_n.non_additive_noise_proba=0.7375"
cmd4=" generator.obs_n.snr=0.3125 generator.inno_n.non_additive_noise_proba=0.86875"
cmd4=" generator.obs_n.snr=0.05 generator.inno_n.non_additive_noise_proba=1"




# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd4="$cmd4 output_dir=$output_dir"
fi

### Increased observation noise

specs=(
"double_common_small"
"double_common_big"
"double_common_small"
"double_common_big"
"double_common_small"
"double_common_big"
"double_common_small"
"double_common_big"
"double_common_small"
"double_common_big"
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
