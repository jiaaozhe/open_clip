# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1
# set the training args

torchrun --nproc_per_node 2 -m open_clip_train.main \
    --name "train_full_finetuning_b32" \
    --save-most-recent \
    --custom-clip-loss \
    --seed 1234 \
    --batch-size 200 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
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
    --model xlm-roberta-large-ViT-H-14 \
    --pretrained "../pretrained_models/openclip-xlm-vit-b32.bin"
