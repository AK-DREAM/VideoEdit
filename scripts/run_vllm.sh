CUDA_VISIBLE_DEVICES=3 vllm serve \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 60000