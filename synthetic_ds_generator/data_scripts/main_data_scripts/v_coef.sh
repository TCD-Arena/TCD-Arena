#!/bin/bash


# Get seed from command line argument, default to 55 if not provided
seed=${1:-55}
# Get output_dir from command line argument, default to empty if not provided
output_dir=${2:-""}



cmd="python generate_dataset.py -m  name="

cmd1=" generator.nonstationary=True nonstationary_change=0.275,0.33125,0.3875,0.44375,0.5" 
cmd2=" n_samples=100 generator.lagged.link_proba=0.075,0.15 generator.instant.link_proba=0.,0.1 seed=$seed"
cmd4=" generator.change_points='[50,100,150,200]'"
cmd5=" generator.change_points='[200,400,600,800]'"

# Add output_dir to command if provided
output_dir_param=""
if [ -n "$output_dir" ]; then
    output_dir_param=" output_dir=$output_dir"
fi


specs=(
 "coef_n_big generator.time_series_n=250$cmd4"
 "coef_n_big generator.time_series_n=1000$cmd5"
 "coef_n_small generator.time_series_n=250$cmd4"
 "coef_n_small generator.time_series_n=1000$cmd5"
)

for spec in "${specs[@]}"; do
    if [[ "$spec" == *"big"* ]]; then
        cmd3=" n_vars=7 generator.lagged.max_lags=4" 
    fi 
    if [[ "$spec" == *"small"* ]]; then
        cmd3=" n_vars=5 generator.lagged.max_lags=3" 
    fi

    echo $cmd$spec$cmd1$cmd2$cmd3$output_dir_param
    eval $cmd$spec$cmd1$cmd2$cmd3$output_dir_param

done




wait
echo "Done"
