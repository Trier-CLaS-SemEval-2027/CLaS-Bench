#!/bin/bash
set -euo pipefail

# Single clean runner for CohereLabs/tiny-aya-global (full bfloat16, early layers 2..12)
# Calls ONLY the original CLaS-Bench Python scripts — zero duplicated Python code.
#
# Usage:
#   bash run_tinyaya.sh              # Run full pipeline (identify -> generate -> eval)
#   bash run_tinyaya.sh identify     # Only Step 1: tokenize CulturaX & compute DiffMean centroids
#   bash run_tinyaya.sh generate     # Only Step 2: DiffMean generation across layers 2,4,6,8,10,12
#   bash run_tinyaya.sh eval         # Only Step 3: FastText forcing + Qwen3-8B judge + plots

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAS_DIR="$SCRIPT_DIR/../../CLaS-Bench"

# vLLM V0 engine is required so forward hooks can attach to model.layers in-process
export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Target model: Cohere2ForCausalLM (3.35B, 36 layers, hidden_size=2048, vocab=262k)
# Uses its own data_tinyaya/ directory because aya-expanse's data_aya/ ([32, 4096]) is incompatible
MODEL="CohereLabs/tiny-aya-global"
MODEL_NAME="${MODEL#*/}"
MODEL_TAG="tinyaya"
SAVE_DIR="data_${MODEL_TAG}"
OUT_ROOT="generation/diffmean_${MODEL_TAG}"

# Balanced 14-language subset (7 High + 7 Low across diverse scripts & families)
# High (7): en, de, es, ar, zh, ja, hi
# Low  (7): cs, uk, el, fa, vi, sw, mt
LANGUAGES=(en de es ar zh ja hi cs uk el fa vi sw mt)
# Full 20-language set (uncomment to run all 10 High + 10 Low):
# LANGUAGES=(ar de en es fr ja zh ru hi ko cs el fa vi uk mt no sw th tr)

# Focus on early layers (Exploration 02 & paper Fig. 15 show Aya steering peaks early)
LAYERS=(2 4 6 8 10 12)
ALPHA="10.0"
SS=$(echo "$ALPHA" | tr '.' '_')

# Step 1: Mirrors CLaS-Bench/identification/0-load-data.sh and 2-diffmean.sh
# Re-tokenizes CulturaX with tiny-aya's tokenizer and averages residual activations per layer
run_identify() {
    cd "$CLAS_DIR/identification"
    echo "=== [Step 1a] Tokenizing CulturaX with $MODEL tokenizer ==="
    python 0-load_data.py --model_name "$MODEL" --save_dir "$SAVE_DIR"

    echo "=== [Step 1b] Computing DiffMean centroids for ${#LANGUAGES[@]} languages ==="
    for lang in "${LANGUAGES[@]}"; do
        [ -f "${SAVE_DIR}/vector.${lang}.${MODEL_TAG}" ] && echo "✓ Skipping $lang (exists)" && continue
        python diffmean.py -m "$MODEL" -l "$lang" -s "${MODEL_TAG} ${MODEL_TAG}"
    done
}

# Step 2: Mirrors CLaS-Bench/2-diffmean-3.sh (uses defaults: bf16, k=70, max_tokens=1024, max_model_len=4096)
run_generate() {
    cd "$CLAS_DIR"
    for L in "${LAYERS[@]}"; do
        echo "=== [Step 2] Generating Layer $L (alpha=$ALPHA, bf16) ==="
        python 2-diffmean_generate.py --batch_mode \
            --output "${OUT_ROOT}/layers/${MODEL_NAME}_layer_${L}_strength_${ALPHA}" \
            --steering_dir "identification/${SAVE_DIR}" \
            --model "$MODEL" \
            --languages "${LANGUAGES[@]}" \
            --layer_start "$L" \
            --layer_end "$((L + 1))" \
            --steering_strength "$ALPHA" \
            --skip_existing
    done
}

# Step 3: Mirrors X-evaluate-language-forcing-diffmean.sh, Y-evaluate-judge-diffmean-0.sh, and B-vis-ablate-diffmean.sh
# Judge scores (semantic relevance) are required alongside FastText forcing to compute the Steering Score
run_eval() {
    cd "$CLAS_DIR"
    for L in "${LAYERS[@]}"; do
        DIR="${OUT_ROOT}/layers/${MODEL_NAME}_layer_${L}_strength_${ALPHA}/strength_${SS}"
        [ -d "$DIR" ] || continue
        echo "=== [Step 3a] FastText forcing: Layer $L ==="
        python evaluate_forcing_success.py --input_dir "$DIR" --output_dir "$DIR"
        if [ "${RUN_JUDGE:-1}" = "1" ]; then
            echo "=== [Step 3b] Qwen3-8B relevance judge: Layer $L ==="
            python judge_llm.py --input_dir "$DIR" --output_dir "$DIR"
        fi
    done

    if [ "${RUN_JUDGE:-1}" = "1" ]; then
        echo "=== [Step 3c] Plotting combined layer ablation ==="
        python vis_ablate_vector.py \
            --input_dirs ${OUT_ROOT}/layers/*/strength_* \
            --output_dir "$OUT_ROOT"
    fi
}

case "$STAGE" in
    identify) run_identify ;;
    generate) run_generate ;;
    eval)     run_eval ;;
    all)      run_identify && run_generate && run_eval ;;
    *)        echo "Usage: bash run_tinyaya.sh [all|identify|generate|eval]"; exit 1 ;;
esac

echo "✓ Done ($STAGE)!"
