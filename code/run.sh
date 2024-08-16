GPU=0
NAME=CRNet
ROOT=/home/root/
DATA_DIR=$ROOT/MagProject/Dataset/
DIR=$ROOT/MagProject/results/
PRETRAIN=$ROOT/MagProject/AlterOpt/pretrain/
BATCHSIZE=200
CR=4

METHOD=Alter
EPOCH=200
STORE_NUM=100
PERIOD=2
SCENARIOS=CDADA

python $ROOT/MagProject/AlterOpt/main.py \
    --epochs $EPOCH \
    --gpu $GPU \
    --root $DIR \
    --name $NAME \
    --data-dir $DATA_DIR \
    --batch-size $BATCHSIZE \
    --workers 0 \
    --cr $CR \
    --method $METHOD \
    --scenarios $SCENARIOS \
    --pretrained $PRETRAIN \
    --store-num $STORE_NUM \
    --period $PERIOD