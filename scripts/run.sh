DATASET=$1
VARIANT=$2
QUANTIZE=$3
COSINE=$4
GUMBLE=$5
CBQKEY=$6
CBRESTART=$7
EIGENQUANTIZER=$8
NAME=$9
ITER=${10}

NAME=$DATASET$VARIANT$NAME
IMPLICIT=True
LOGS='/vol/biomedic3/agk21/testEigenSlots2/LOGS-LEMMA2'
echo "========================IMPLICIT=============================="
python /vol/biomedic3/agk21/testEigenSlots2/object_discovery.py \
                                            --dataset_name $DATASET \
                                            --variant $VARIANT \
                                            --exp_name $NAME \
                                            --batch_size 16 \
                                            --model_dir $LOGS \
                                            --learning_rate 0.0004 \
                                            --quantize $QUANTIZE \
                                            --cosine $COSINE \
                                            --gumble $GUMBLE \
                                            --num_iterations $ITER \
                                            --cb_qk $CBQKEY \
                                            --implicit $IMPLICIT \
                                            --eigen_quantizer $EIGENQUANTIZER \
                                            --restart_cbstats $CBRESTART



