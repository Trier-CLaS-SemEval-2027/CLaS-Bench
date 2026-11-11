#!/bin/bash

pip install vllm bitsandbytes datasets

SCRIPT="1-prompt_generate.py"
MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b
MODEL_NAME=${MODEL#*/}

export VLLM_USE_V1=0
# export VLLM_FLASH_ATTN_VERSION=2 # for gemma-2

OUTPUT="generation/baseline"
LANGUAGES="en de es"

echo "Running baseline experiments"
echo "Model: $MODEL"
echo "Languages: $LANGUAGES"
echo "Output: $OUTPUT"
echo "======================================"

python $SCRIPT --batch_mode \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --languages $LANGUAGES \
    --instruction_mode "language_instruction" \
    --max_model_len 2048 \
    --quantization bitsandbytes \
    --dtype "float16"


