
python Claim_Generation.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_train.processed.json \
    --entity_dict ./data/entity_dict.json \
    --QA_path ./data/precompute_QAs.json \
    --QA2D_model_path ../QA2D/outputs/best_model \
    --sense_to_vec_path ../dependencies/s2v_old \
    --save_path ../New_Results/fast_NEI_extend_100000_all.json \
    --gpu_index 1 \
    --range_start 100000 \
    --range_end -1 \
    --claim_type NEI