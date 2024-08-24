GPU=0
NAME=CRNet
ROOT=/teams/ius_1663576043/zxd/MagProject/
DATA_DIR=$ROOT/3GPP
DIR=$ROOT/results/
PRETRAIN=$ROOT/AlterOpt-main/code/pretrain/
BATCHSIZE=100
CR=4

METHOD=Alter
EPOCH=200
STORE_NUM=100
PERIOD=2
SCENARIOS=CDADA

python $ROOT/AlterOpt-main/code/main.py \
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

