#!/usr/bin/env bash
set -x

PL_mode_arr=('-' 'PLepi' 'PLePI')
label_quality_arr=('LQ' 'HQ')
work_dir='./experiments/ablation/temp/'
gpu_id="6"


ngs_match_arr=()
cell_recovery_rate_arr=()


for PL_mode in "${PL_mode_arr[@]}"
do
    for label_quality in "${label_quality_arr[@]}"
    do
        python ./experiments/ablation/run_exp.py --PL_mode $PL_mode --label_quality $label_quality --work_dir $work_dir --gpu_id $gpu_id

    done
done
