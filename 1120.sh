# finetune.py
WORLD_SIZE=8  torchrun --nproc_per_node=8 --master_port=3192 finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'math_data.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 32 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora

WORLD_SIZE=6  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --master_port=3192 finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'math_50k.json' \
  --output_dir './trained_models/llama-lora-50k' \
  --batch_size 24 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 0.2 \
  --adapter_name lora

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=3192 finetune.py   \
    --base_model 'yahma/llama-7b-hf'   \
    --data_path 'math_10k.json'  \
    --output_dir './trained_models/llama-lora-10k/' \
    --batch_size 16  \
    --micro_batch_size 4   \
    --num_epochs 3   \
    --learning_rate 3e-4   \
    --cutoff_len 256   \
    --val_set_size 120 \
    --eval_step 80 \
    --save_step 80  \
    --adapter_name lora

# evaluation
## 1. for math_data.json finetuned model
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'

CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
### COT
CUDA_VISIBLE_DEVICES=4 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora' \
    --use_CoT \
    --save-suffix '_CoT'

CUDA_VISIBLE_DEVICES=5 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora' \
    --use_CoT \
    --save-suffix '_CoT'

CUDA_VISIBLE_DEVICES=6 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora' \
    --use_CoT \
    --save-suffix '_CoT'

## 2. for math_50k.json finetuned model
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-50k' \
    --save-suffix 'ft50k'

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-50k' \
    --save-suffix 'ft50k'

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-50k' \
    --save-suffix 'ft50k'
### COT
CUDA_VISIBLE_DEVICES=7 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-50k' \
    --use_CoT \
    --save-suffix 'ft50k_CoT' 
# TODO
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-50k' \
    --use_CoT \
    --save-suffix 'ft50k_CoT' 

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-50k' \
    --use_CoT \
    --save-suffix 'ft50k_CoT' 
# END TODO

## 3. for math_10k.json finetuned model
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-10k' \
    --save-suffix 'ft10k'

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-10k' \
    --save-suffix 'ft10k'

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-10k' \
    --save-suffix 'ft10k'
### COT
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-10k' \
    --use_CoT \
    --save-suffix 'ft10k_CoT' 
CUDA_VISIBLE_DEVICES=4 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-10k' \
    --use_CoT \
    --save-suffix 'ft10k_CoT' 

CUDA_VISIBLE_DEVICES=5 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora-10k' \
    --use_CoT \
    --save-suffix 'ft10k_CoT' 
    
# without finetune
# zero-shot task 不行，模型不懂格式，输出都是错的（不停重复）
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --model LLaMA-7B \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' 

CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --model LLaMA-7B \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' 