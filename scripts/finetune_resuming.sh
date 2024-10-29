python -m open_clip_train.main \
    --train-data="./dataset/train.jsonl" \
    --val-data="./dataset/test.jsonl"  \
    --resume "../logs/train_full_finetuning_b32/checkpoints/epoch_last.pt"