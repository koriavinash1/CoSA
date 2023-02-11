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
LOGS='/vol/biomedic3/agk21/testEigenSlots2/LOGSBenchmarkingMultidataset'
python /vol/biomedic3/agk21/testEigenSlots2/train.py \
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
                                            --eigen_quantizer $EIGENQUANTIZER \
                                            --restart_cbstats $CBRESTART



# sample script: Benckmark exp.
# ./run.sh 8 4 64 0.99 8 0.0 True False False True False False False False bnmtest 3