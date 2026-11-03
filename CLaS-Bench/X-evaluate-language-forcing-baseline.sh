#!/bin/bash

pip install fasttext-numpy2-wheel seaborn matplotlib

SCRIPT="evaluate_forcing_success.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}

INPUT="generation/baseline/language_instruction/"

python $SCRIPT \
        --input_path $INPUT \
        --output_path $INPUT


