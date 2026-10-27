#!/bin/bash

# Experiment Script for Propensity Misspecification (Bias Injection) with Parallel Execution
# Dataset: Ali-CCP
# Loss: IPS
# Methods: Base IPS, APL-Heuristic, APL-KL, APL-Heuristic-KL, APL-MC-Dropout, APL-MC-Dropout-KL

# --- Job Control ---
# GPUs to use (comma-separated, e.g., "2,3")
GPUS=(2 3)
NUM_GPUS=${#GPUS[@]}

# Maximum concurrent jobs
MAX_JOBS=6

# Bias Parameters
BIAS_SCALES=(0.8 0.9 1.0 1.1 1.2)
BIAS_SHIFTS=(-0.2 -0.1 0.0 0.1 0.2)
BIAS_NOISES=(0.0 0.05 0.1 0.15 0.2)
ENABLE_META=0
META_LR=0.01
META_UPDATE_INTERVAL=2
META_CLIP_NORM=1.0
META_INNER_LR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs/bias_injection"
mkdir -p "$LOG_DIR"

# Function to manage job concurrency
job_control() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Function to run a single experiment
run_single_exp() {
    local bias_type=$1
    local bias_value=$2
    local log_suffix=$3
    local extra_args=$4
    local method_name=$5
    local method_args=$6
    local gpu_id=$7
    
    local log_file="${LOG_DIR}/aliccp_ips_${method_name}_${log_suffix}.log"

    # Check if log file exists and is not empty (resume capability)
    if [ -s "$log_file" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping $method_name ($bias_type=$bias_value) - Log exists: $log_file"
        return
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $method_name ($bias_type=$bias_value) on GPU $gpu_id"
    
    meta_args=""
    if [[ "${ENABLE_META}" -eq 1 ]]; then
        meta_args="--apl_learnable --meta_lr ${META_LR} --meta_update_interval ${META_UPDATE_INTERVAL} --meta_clip_norm ${META_CLIP_NORM}"
        if [[ -n "${META_INNER_LR}" ]]; then
            meta_args="${meta_args} --meta_inner_lr ${META_INNER_LR}"
        fi
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id python ${SCRIPT_DIR}/run_bias_experiments.py \
        --loss_type ips \
        $meta_args \
        $method_args \
        $extra_args \
        > "$log_file" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $method_name ($bias_type=$bias_value)"
}

# --- Experiment Queue ---
count=0

dispatch_job() {
    local bias_type=$1
    local bias_value=$2
    local log_suffix=$3
    local extra_args=$4
    local method_name=$5
    local method_args=$6

    job_control
    
    # Assign GPU in round-robin fashion
    local gpu_idx=$((count % NUM_GPUS))
    local gpu_id=${GPUS[$gpu_idx]}
    
    # Run in background
    run_single_exp "$bias_type" "$bias_value" "$log_suffix" "$extra_args" "$method_name" "$method_args" "$gpu_id" &
    
    count=$((count + 1))
}

# Function to run all methods for a given bias setting
run_all_methods() {
    local bias_type=$1
    local bias_value=$2
    local log_suffix=$3
    local extra_args=$4

    # 1. Base IPS
    dispatch_job "$bias_type" "$bias_value" "${log_suffix}" "$extra_args" "base" ""

    # 2. APL-Heuristic
    dispatch_job "$bias_type" "$bias_value" "${log_suffix}" "$extra_args" "heuristic" "--apl_method heuristic --apl_alpha 1.0"

    # 3. APL-KL
    dispatch_job "$bias_type" "$bias_value" "${log_suffix}" "$extra_args" "kl" "--apl_method kl --apl_lambda 1.0"

    # 4. APL-Heuristic-KL
    dispatch_job "$bias_type" "$bias_value" "${log_suffix}" "$extra_args" "heuristic_kl" "--apl_method heuristic_kl --apl_alpha 1.0 --apl_lambda 1.0"

    # 5. APL-MC-Dropout
    dispatch_job "$bias_type" "$bias_value" "${log_suffix}" "$extra_args" "mc_dropout" "--apl_method mc_dropout --apl_alpha 1.0 --mcdp_n 5"

    # 6. APL-MC-Dropout-KL
    dispatch_job "$bias_type" "$bias_value" "${log_suffix}" "$extra_args" "mc_dropout_kl" "--apl_method mc_dropout_kl --apl_alpha 1.0 --apl_lambda 1.0 --mcdp_n 5"
}

# --- Main Execution Loop ---

# 1. Scaling Experiments
for scale in "${BIAS_SCALES[@]}"; do
    run_all_methods "Scaling" "$scale" "scale${scale}" "--propensity_scale $scale"
done

# 2. Shift Experiments
for shift in "${BIAS_SHIFTS[@]}"; do
    run_all_methods "Shift" "$shift" "shift${shift}" "--propensity_shift $shift"
done

# 3. Noise Experiments
for noise in "${BIAS_NOISES[@]}"; do
    run_all_methods "Noise" "$noise" "noise${noise}" "--propensity_noise $noise"
done

wait
echo "All parallel bias experiments completed."
