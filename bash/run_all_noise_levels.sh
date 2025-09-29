#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_results_dir>"
    exit 1
fi

SNRS=(-25 -20 -15 -10 -5 0)
SEEDS=(1 2 3 4 5)
RESULTS_DIRPATH="$1"

for SNR in "${SNRS[@]}"; do
    mkdir -p "${RESULTS_DIRPATH}/noise_level_${SNR}"
    for SEED in "${SEEDS[@]}"; do
        echo "Running SNR of $SNR with seed=$SEED"
        python src/prospero/runners/run_noisy.py --snr $SNR --seed $SEED --results_dirpath "$RESULTS_DIRPATH" --full_deterministic
    done
    python src/prospero/runners/etl_results.py --task noise_level_${SNR} --results_dirpath "$RESULTS_DIRPATH"
done