# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=4,5,6,7
# set the training args

torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-most-recent \
    --custom-clip-loss \
    --force-patch-dropout 0.0 \
    --seed 1234 \
    --batch-size 100 \
    --precision amp \
    --workers 2 \
    --report-to tensorboard \
    --save-frequency 5 \
    --dataset-type jsonl \
    --csv-separator="," \
    --train-data "./dataset/train.jsonl"\
    --val-data "./dataset/test.jsonl" \
    --csv-img-key image_path \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=32 \
    --model xlm-roberta-base-ViT-B-32 \
    --pretrained "../pretrained_models/openclip-xlm-vit-b32.bin"
