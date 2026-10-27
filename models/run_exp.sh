#!/bin/bash

# Run all APL experiments on Ali-CCP and AliExpress datasets
# GPU: 1, 3 (2 GPUs)
# Max parallel: 8 tasks
# Skip already running/completed experiments
# Ablation experiments only on Ali-CCP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Logs will be placed in ../logs/main or ../logs/ablation
LOG_BASE_DIR="${SCRIPT_DIR}/../logs"
PYTHON="python"

mkdir -p "${LOG_BASE_DIR}/main"
mkdir -p "${LOG_BASE_DIR}/ablation"

LOSS_TYPES=('ips' 'doubly_robust' 'escm2_ips' 'escm2_dr')
REGIONS=('ES' 'FR' 'NL' 'US')
APL_METHODS=('None' 'heuristic' 'mc_dropout' 'kl' 'heuristic_kl' 'mc_dropout_kl')
MCDP_N=5
GPUS=(1 3)
MAX_PARALLEL=16

echo "=========================================="
echo "Running all APL experiments"
echo "GPUs: ${GPUS[@]}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Log base directory: ${LOG_BASE_DIR}"
echo "Ablation: Ali-CCP only"
echo "=========================================="

get_exp_name() {
    local dataset=$1
    local region=$2
    local loss_type=$3
    local apl_method=$4
    local apl_alpha=$5
    local apl_lambda=$6
    local mcdp_n=$7
    
    local exp_name="${dataset}"
    if [[ -n "$region" && "$region" != "None" ]]; then
        exp_name+="_${region}"
    fi
    exp_name+="_${loss_type}"
    if [[ -n "$apl_method" && "$apl_method" != "None" ]]; then
        exp_name+="_${apl_method}"
    fi
    if [[ -n "$apl_alpha" && "$apl_alpha" != "None" ]]; then
        exp_name+="_alpha${apl_alpha}"
    fi
    if [[ -n "$apl_lambda" && "$apl_lambda" != "None" ]]; then
        exp_name+="_lambda${apl_lambda}"
    fi
    if [[ -n "$mcdp_n" && "$mcdp_n" != "None" ]]; then
        exp_name+="_mcdp${mcdp_n}"
    fi
    echo "$exp_name"
}

run_experiment() {
    local dataset=$1
    local region=$2
    local loss_type=$3
    local apl_method=$4
    local apl_alpha=$5
    local apl_lambda=$6
    local mcdp_n=$7
    local gpu=$8
    
    local exp_name=$(get_exp_name "$dataset" "$region" "$loss_type" "$apl_method" "$apl_alpha" "$apl_lambda" "$mcdp_n")
    
    # Determine log directory (main or ablation)
    local target_log_dir="${LOG_BASE_DIR}/main"
    if [[ -n "$apl_alpha" && "$apl_alpha" != "None" && "$apl_alpha" != "1.0" ]]; then
        target_log_dir="${LOG_BASE_DIR}/ablation"
    elif [[ -n "$apl_lambda" && "$apl_lambda" != "None" && "$apl_lambda" != "1.0" ]]; then
        target_log_dir="${LOG_BASE_DIR}/ablation"
    fi
    
    local log_file="${target_log_dir}/${exp_name}.log"
    
    # Skip if log file exists and is not empty (already running or completed)
    if [[ -s "${log_file}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping ${exp_name}: log already exists" | tee -a "${LOG_BASE_DIR}/run.log"
        return 0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting on GPU ${gpu}: ${exp_name}" | tee -a "${LOG_BASE_DIR}/run.log"
    
    local cmd="CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} ${SCRIPT_DIR}/mmoe_${dataset}.py --loss_type ${loss_type}"
    if [[ -n "$region" && "$region" != "None" ]]; then
        cmd+=" --region ${region}"
    fi
    if [[ -n "$apl_method" && "$apl_method" != "None" ]]; then
        cmd+=" --apl_method ${apl_method}"
    fi
    if [[ -n "$apl_alpha" && "$apl_alpha" != "None" ]]; then
        cmd+=" --apl_alpha ${apl_alpha}"
    fi
    if [[ -n "$apl_lambda" && "$apl_lambda" != "None" ]]; then
        cmd+=" --apl_lambda ${apl_lambda}"
    fi
    if [[ -n "$mcdp_n" && "$mcdp_n" != "None" ]]; then
        cmd+=" --mcdp_n ${mcdp_n}"
    fi
    cmd+=" > ${log_file} 2>&1 &"
    
    eval "${cmd}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched on GPU ${gpu}: ${exp_name} (PID: $!)" | tee -a "${LOG_BASE_DIR}/run.log"
}

export -f run_experiment get_exp_name
export SCRIPT_DIR LOG_BASE_DIR PYTHON MCDP_N

TASKS=()

# ===== Main experiments (ID 1-120) =====
echo "Generating main experiments..."

# Ali-CCP main experiments
for loss_type in "${LOSS_TYPES[@]}"; do
    for apl_method in "${APL_METHODS[@]}"; do
        if [[ "$apl_method" == "None" ]]; then
            TASKS+=("aliccp None ${loss_type} None None None None")
        elif [[ "$apl_method" == "heuristic" ]]; then
            TASKS+=("aliccp None ${loss_type} heuristic 1.0 None None")
        elif [[ "$apl_method" == "mc_dropout" ]]; then
            TASKS+=("aliccp None ${loss_type} mc_dropout 1.0 None ${MCDP_N}")
        elif [[ "$apl_method" == "kl" ]]; then
            TASKS+=("aliccp None ${loss_type} kl None 1.0 None")
        elif [[ "$apl_method" == "heuristic_kl" ]]; then
            TASKS+=("aliccp None ${loss_type} heuristic_kl 1.0 1.0 None")
        elif [[ "$apl_method" == "mc_dropout_kl" ]]; then
            TASKS+=("aliccp None ${loss_type} mc_dropout_kl 1.0 1.0 ${MCDP_N}")
        fi
    done
done

# AliExpress main experiments
for region in "${REGIONS[@]}"; do
    for loss_type in "${LOSS_TYPES[@]}"; do
        for apl_method in "${APL_METHODS[@]}"; do
            if [[ "$apl_method" == "None" ]]; then
                TASKS+=("aliexpress ${region} ${loss_type} None None None None")
            elif [[ "$apl_method" == "heuristic" ]]; then
                TASKS+=("aliexpress ${region} ${loss_type} heuristic 1.0 None None")
            elif [[ "$apl_method" == "mc_dropout" ]]; then
                TASKS+=("aliexpress ${region} ${loss_type} mc_dropout 1.0 None ${MCDP_N}")
            elif [[ "$apl_method" == "kl" ]]; then
                TASKS+=("aliexpress ${region} ${loss_type} kl None 1.0 None")
            elif [[ "$apl_method" == "heuristic_kl" ]]; then
                TASKS+=("aliexpress ${region} ${loss_type} heuristic_kl 1.0 1.0 None")
            elif [[ "$apl_method" == "mc_dropout_kl" ]]; then
                TASKS+=("aliexpress ${region} ${loss_type} mc_dropout_kl 1.0 1.0 ${MCDP_N}")
            fi
        done
    done
done

# ===== Ablation: apl_alpha (Ali-CCP only, ID 121-152) =====
echo "Generating apl_alpha ablation experiments (Ali-CCP only)..."

ALPHA_ABLATION_VALUES=(0.0 0.5 1.5 2.0)

for apl_alpha in "${ALPHA_ABLATION_VALUES[@]}"; do
    for loss_type in "${LOSS_TYPES[@]}"; do
        TASKS+=("aliccp None ${loss_type} heuristic ${apl_alpha} None None")
        TASKS+=("aliccp None ${loss_type} mc_dropout ${apl_alpha} None ${MCDP_N}")
    done
done

# ===== Ablation: apl_lambda (Ali-CCP only, ID 153-168) =====
echo "Generating apl_lambda ablation experiments (Ali-CCP only)..."

LAMBDA_ABLATION_VALUES=(0.1 0.5 5.0 10.0)

for apl_lambda in "${LAMBDA_ABLATION_VALUES[@]}"; do
    for loss_type in "${LOSS_TYPES[@]}"; do
        TASKS+=("aliccp None ${loss_type} kl None ${apl_lambda} None")
    done
done

echo "Total experiments: ${#TASKS[@]}"
echo ""

# Assign GPUs round-robin
for i in "${!TASKS[@]}"; do
    gpu_idx=$((i % ${#GPUS[@]}))
    gpu=${GPUS[$gpu_idx]}
    TASKS[$i]="${TASKS[$i]} ${gpu}"
done

# Launch tasks with shared max parallel control
echo "Launching experiments..."
for task in "${TASKS[@]}"; do
    # Check current running processes (shared with other instances)
    running=$(ps aux | grep "mmoe_ali" | grep -v grep | wc -l)
    
    # Wait if max parallel reached
    while [[ $running -ge $MAX_PARALLEL ]]; do
        sleep 10
        running=$(ps aux | grep "mmoe_ali" | grep -v grep | wc -l)
    done
    
    run_experiment $task
    sleep 0.5
done

# Wait for all background jobs to complete
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Check logs in: ${LOG_BASE_DIR}"
echo "Main log: ${LOG_BASE_DIR}/run.log"
echo "=========================================="
