#!/bin/bash

# This generates all observational noise datasets
# For the sets with 7 variables, a bigger max lag is used.

# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}


cmd="python generate_dataset.py -m  name="
cmd2=" generator.time_series_n=1000 seed=$seed n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1"
cmd3=" generator.lagged.alternativ_parameter_range='[0.8,0.9]'  generator.remove_n_variables_for_confounding=1"

# Add output_dir to command if provided
if [ -n "$output_dir" ]; then
    cmd3="$cmd3 output_dir=$output_dir"
fi

specs=()

for x in 0.135,0.27625,0.4175,0.55875,0.7; do
    specs+=("conf_l_7_2 n_vars=8 generator.lagged.alternative_link_proba=${x} generator.lagged.alternative_coeff_ts=1 generator.lagged.max_lags=2 ")
done

for x in 0.135,0.27625,0.4175,0.55875,0.7; do
    specs+=("conf_l_3_2 n_vars=4 generator.lagged.alternative_link_proba=${x} generator.lagged.alternative_coeff_ts=1 generator.lagged.max_lags=2 ")
done

for x in 0.135,0.27625,0.4175,0.55875,0.7; do
    specs+=("conf_l_3_4 n_vars=4 generator.lagged.alternative_link_proba=${x} generator.lagged.alternative_coeff_ts=1 generator.lagged.max_lags=4 ")
done





for spec in "${specs[@]}"; do
    echo $cmd$spec$cmds2$cmd3 &
    eval $cmd$spec$cmd2 $cmd3
done

wait
echo "Done"
