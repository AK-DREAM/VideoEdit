#!/bin/bash

if [ "$CONDA_DEFAULT_ENV" != "vllm" ]; then
    echo "当前环境: $CONDA_DEFAULT_ENV"
    echo "正在切换到vllm环境..."
    eval "$(conda shell.bash hook)"
    conda activate vllm
    echo "✓ 已切换到vllm环境"
else
    echo "✓ 已在vllm环境中"
fi

CUDA_VISIBLE_DEVICES=3  vllm serve \
    ./weights/Qwen/Qwen3-VL-8B-Thinking \
    --quantization FP8 \
    --allowed-local-media-path /home/keli/VideoEdit/\
    --reasoning-parser deepseek_r1 \
    --trust-remote-code \
    --max-model-len 128000 \
    --enable-prefix-caching \
    --port 8080