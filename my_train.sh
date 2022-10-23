# ./distributed_train.sh 8 /home/ubuntu/imagenet --model twins_pcpvt_small_spike \
# --sched cosine --epochs 160 --warmup-epochs 5 --lr 0.0006 --opt AdamW \
# --reprob 0.2 --remode pixel --batch-size 48 --log-interval 300 --initial-checkpoint 'output/train/20221020-214727-twins_pcpvt_small_spike-224/model_best.pth.tar' \
# --amp -j 8 --pin-mem --log-wandb --clip-grad 5.0

./distributed_train.sh 6 /root/autodl-tmp/imagenet --model twins_pcpvt_small_spike_v2 \
--sched cosine --epochs 160 --warmup-epochs 5 --lr 0.0006 --opt AdamW \
--reprob 0.2 --remode pixel --batch-size 64 --log-interval 300 \
--amp -j 6 --pin-mem --log-wandb --clip-grad 5.0