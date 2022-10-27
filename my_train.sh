# ./distributed_train.sh 8 /home/ubuntu/imagenet --model twins_pcpvt_small_spike \
# --sched cosine --epochs 160 --warmup-epochs 5 --lr 0.0003 --opt AdamW \
# --reprob 0.2 --remode pixel --batch-size 48 --log-interval 300 --initial-checkpoint 'output/train/20221023-085603-twins_pcpvt_small_spike_v2-224/model_best.pth.tar' \
# --amp -j 8 --pin-mem --log-wandb --clip-grad 5.0

# ./distributed_train.sh 8 /home/ubuntu/imagenet --model twins_pcpvt_small_spike \
# --sched cosine --epochs 60 --warmup-epochs 5 --lr 0.0006 --opt AdamW --weight-decay 0.005 \
# --reprob 0.2 --remode pixel --batch-size 48 --log-interval 300 --initial-checkpoint 'output/train/20221021-192834-twins_pcpvt_small_spike-224/checkpoint-152.pth.tar' \
# --amp -j 8 --pin-mem --log-wandb --clip-grad 5.0

# ./distributed_train.sh 8 /home/ubuntu/imagenet --model twins_pcpvt_small_spike \
# --sched cosine --epochs 60 --warmup-epochs 5 --lr 0.003 --opt AdamW --weight-decay 0.00005 \
# --reprob 0.25 --remode pixel --batch-size 48 --log-interval 300 --initial-checkpoint 'output/train/20221021-192834-twins_pcpvt_small_spike-224/checkpoint-152.pth.tar' \
# --amp -j 8 --pin-mem --clip-grad 5.0 --drop-path 0.3 --mixup 0.8 --cutmix 1.0 --aa 'rand-m9-mstd0.5-inc1' --train-interpolation 'bicubic'

# python inference.py /home/ubuntu/imagenet/val/ --model twins_pcpvt_small_spike --checkpoint output/train/20221021-192834-twins_pcpvt_small_spike-224/model_best.pth.tar

./distributed_train.sh 6 /root/autodl-tmp/imagenet --model twins_pcpvt_small_spike_v2 \
--sched cosine --epochs 300 --warmup-epochs 5 --lr 0.0015 --opt AdamW --weight-decay 0.05 \
--reprob 0.25 --remode pixel --batch-size 256 --log-interval 300 \
--amp -j 6 --pin-mem --log-wandb --clip-grad 5.0 --drop-path 0.1 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 10 --opt-eps 1e-8 --aa 'rand-m9-mstd0.5-inc1' --train-interpolation 'bicubic'

# ./distributed_train.sh 6 /root/autodl-tmp/imagenet --model twins_pcpvt_small_spike_v2 \
# --sched cosine --epochs 160 --warmup-epochs 5 --lr 0.0003 --opt AdamW --opt-eps 1e-8 \
# --reprob 0.2 --remode pixel --batch-size 256 --log-interval 300 --initial-checkpoint 'output/train/20221023-085603-twins_pcpvt_small_spike_v2-224/model_best.pth.tar' \
# --amp -j 6 --pin-mem --log-wandb --clip-grad 5.0