#!/bin/bash

pip install fasttext-numpy2-wheel seaborn matplotlib

SCRIPT="evaluate_forcing_success.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}

for LAYER_START in 4; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 10.0; do
        SS=$(echo "$STEERING_STRENGTH" | tr '.' '_')
        INPUT="generation/diffmean/layers/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}/strength_${SS}"

        echo "Running steering evaluation"
        echo "Layers: $LAYER_START to $((LAYER_END-1)), Strength: $STEERING_STRENGTH"

        python $SCRIPT \
            --input_path $INPUT \
            --output_path $INPUT
    done
done
