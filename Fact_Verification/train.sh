export CUDA_VISIBLE_DEVICES=4

python3 run_hover.py \
--dataset_name FEVER \
--model_type roberta \
--model_name_or_path roberta-large \
--sub_task claim_verification \
--do_train \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--learning_rate 1e-5 \
--num_train_epochs 5.0 \
--evaluate_during_training \
--max_seq_length 200  \
--max_query_length 60 \
--gradient_accumulation_steps 2  \
--max_steps 20000 \
--save_steps 1000 \
--logging_steps 1000 \
--overwrite_cache \
--num_labels 3 \
--data_dir ../data/ \
--train_file fever_generated_claims.json \
--predict_file fever_dev.processed.json \
--output_dir ./output/roberta_zero_shot \