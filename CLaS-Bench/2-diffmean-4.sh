#!/bin/bash

pip install vllm bitsandbytes datasets

SCRIPT="2-diffmean_generate.py"
MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b

MODEL_NAME=${MODEL#*/}
STEERING_DIR="identification/data_aya" 
LANGUAGES="en de es"

export VLLM_USE_V1=0
# export VLLM_FLASH_ATTN_VERSION=2 # for gemma-2

# Steering parameters - test representative layers across the network (early, early-mid, late-mid, late)
for LAYER_START in 4 12 20 28; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 10.0; do
        OUTPUT="generation/diffmean/layers/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}"

        echo "Running steering intervention"
        echo "Layers: $LAYER_START to $((LAYER_END-1)), Strength: $STEERING_STRENGTH"
        echo "Languages: $LANGUAGES"
        echo "Output path: $OUTPUT"

        python $SCRIPT --batch_mode \
            --output "$OUTPUT" \
            --steering_dir "$STEERING_DIR" \
            --model "$MODEL" \
            --languages $LANGUAGES \
            --layer_start $LAYER_START \
            --layer_end $LAYER_END \
            --steering_strength $STEERING_STRENGTH \
            --max_model_len 2048 \
            --quantization bitsandbytes \
            --skip_existing
    done
done


