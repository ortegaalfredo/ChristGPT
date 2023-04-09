# Example command to finetune the llama-13-hf base model on 2xGPUs with the ChristGPT dataset

OMP_NUM_THREADS=4 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune.py     --base_model '/llama-13b-hf/'     --data_path 'dataset.json'     --output_dir './lora-alpaca'     --num_epochs=10     --cutoff_len=512     --group_by_length     --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'     --lora_r=16     --micro_batch_size=8 
