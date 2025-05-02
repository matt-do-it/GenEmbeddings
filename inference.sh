#!/bin/zsh

mlx_lm.generate \
    --model ./models/gemma3-1b-it/transformers \
    --adapter-path ./adapters/gemma3-1b-it/mlx \
	--prompt "The stock market will go up in..." \
	--ignore-chat-template \
	--verbose T