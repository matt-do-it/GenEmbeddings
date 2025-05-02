#!/bin/zsh

mlx_lm.fuse \
    --model models/gemma3-1b-it/transformers \
    --adapter-path ./adapters/gemma3-1b-it/mlx \
    --save-path ./fused/gemma3-1b-it/transformers
