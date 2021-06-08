
python Claim_Generation.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_train.processed.json \
    --entity_dict ../output/intermediate/entity_dict_train.json \
    --QA_path ./output/intermediate/precompute_QAs_train.json \
    --QA2D_model_path ../QA2D_model \
    --sense_to_vec_path ../dependencies/s2v_old \
    --save_path ../output/NEI_claims.json \
    --range_start 0 \
    --range_end -1 \
    --claim_type NEI