python /home/AlterOpt-main/code/main.py \
    --epochs 200 \
    --gpu 0 \
    --root '/home/AlterOpt/' \
    --name 'CRNet' \
    --data-dir '/home/AlterOpt/Dataset/' \
    --batch-size 100 \
    --workers 0 \
    --cr 4 \
    --method 'Alter' \
    --scenarios 'CDADA' \
    --pretrained '/home/AlterOpt/code/pretrain/ \
    --store-num 100 \
    --period 2

