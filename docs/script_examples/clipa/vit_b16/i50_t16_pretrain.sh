torchrun --nproc_per_node 8 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data '/path/to/laion-400m' \
    --dataset-type webdataset \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.2 \
    --batch-size 8192 \
    --aug-cfg scale='(0.4, 1.0)' \
    --epochs 6 \
    --workers 6 \
    --model ViT-B-16-CL16 \
    --precision 'amp_bf16' \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --force-image-size 112 \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val '/path/to/imagenet/val'