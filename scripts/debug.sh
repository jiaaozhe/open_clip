# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1
cd /home/yajun/projects/jiaaozhe/open_clip/src
# set the training args

torchrun --nproc_per_node 2 -m open_clip_train.main \
    --save-most-recent \
    --custom-clip-loss \
    --delete-previous-checkpoint \
    --lock-image \
    --lock-text \
    --seed 1234 \
    --batch-size 1 \
    --precision amp \
    --workers 2 \
    --report-to tensorboard \
    --save-frequency 5 \
    --dataset-type csv \
    --csv-separator="," \
    --train-data "/home/yajun/projects/jiaaozhe/open_clip/src/dataset/train.csv"\
    --val-data "/home/yajun/projects/jiaaozhe/open_clip/src/dataset/train.csv" \
    --csv-img-key image_path \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=200 \
    --model xlm-roberta-large-ViT-H-14 \
    --pretrained "/home/yajun/projects/jiaaozhe/open_clip/pretrained_models/open_clip_pytorch_model.bin"
