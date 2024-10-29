# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1
# set the training args

torchrun --nproc_per_node 2 -m open_clip_train.main \
    --save-most-recent \
    --delete-previous-checkpoint \
    --lock-image \
    --lock-text \
    --seed 1234 \
    --batch-size 1 \
    --precision amp \
    --workers 2 \
    --report-to tensorboard \
    --save-frequency 5 \
    --dataset-type jsonl \
    --csv-separator="," \
    --train-data "./dataset/train.jsonl"\
    --val-data "./dataset/train.jsonl" \
    --csv-img-key image_path \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=200 \
    --model xlm-roberta-large-ViT-H-14 \
    --pretrained "../pretrained_models/openclip-xlm-vit-h14.bin"
