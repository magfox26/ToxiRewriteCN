NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
--model /home/ToxiRewriteCN/models/Meta-Llama-3.1-8B-Instruct \ 
--model_type llama3_1 \
--train_type full \
--dataset /home/ToxiRewriteCN/finetuning_llama3-8b/data/r1_train.json \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--learning_rate 1e-6 \
--lr_scheduler_type cosine \
--eval_strategy "epoch" \
--gradient_accumulation_steps 2 \
--save_total_limit 1 \
--warmup_ratio 0.05 \
--logging_steps 1 \
--max_length 32768 \
--weight_decay 1e-4 \
--deepspeed zero3 \
--dataloader_num_workers 4 \
--output_dir /home/ToxiRewriteCN/finetuning_llama3-8b/output/llama3_8b_r1 \
--report_to swanlab \
--swanlab_token your_own_swanlab_token 
