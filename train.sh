#!/bin/zsh

mlx_lm.lora \
    --model ./models/gemma3-1b-it/transformers \
    --train \
    --data ./training/ \
    --batch-size 1 \
    --adapter-path ./adapters/gemma3-1b-it/mlx \
    --iters 600