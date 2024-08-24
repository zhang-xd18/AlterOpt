python /home/code/main.py \
    --epochs 200 \
    --gpu 0 \
    --root '/home/results/' \
    --name 'CRNet' \
    --data-dir '/home/data/' \
    --batch-size 100 \
    --workers 0 \
    --cr 4 \
    --method 'Alter' \
    --scenarios 'CDADA' \
    --pretrained '/home/code/pretrain/' \
    --store-num 100 \
    --period 2

