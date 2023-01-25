ENCODERRES=$1
DECODERRES=$2
IMSIZE=$3
CBDECAY=$4
NCONCEPTS=$5
OPWEIGHTAGE=$6
NOPOSITION=${7}
QUANTIZE=$8
COSINE=$9
VAR=${10}
CBQKEY=${11}
CBRESTART=${12}
EIGENQUANTIZER=${13}
BINARIZE=${14}
NAME=${15}
ITER=${16}

LOGS='/vol/biomedic3/agk21/testEigenSlots2/Testingsss'
python /vol/biomedic3/agk21/testEigenSlots2/train.py \
                                            --exp_name $NAME \
                                            --batch_size 16 \
                                            --model_dir $LOGS \
                                            --img_size $IMSIZE \
                                            --encoder_res $ENCODERRES \
                                            --decoder_res $DECODERRES \
                                            --learning_rate 0.001 \
                                            --cb_decay $CBDECAY \
                                            --nunique_objects $NCONCEPTS \
                                            --max_slots $((2*$NCONCEPTS + 2)) \
                                            --overlap_weightage $OPWEIGHTAGE \
                                            --quantize $QUANTIZE \
                                            --cosine $COSINE \
                                            --binarize $BINARIZE \
                                            --eigen_noposition $NOPOSITION \
                                            --variational $VAR \
                                            --num_iterations $ITER \
                                            --cb_qk $CBQKEY \
                                            --eigen_quantizer $EIGENQUANTIZER \
                                            --restart_cbstats $CBRESTART



# sample script: Benckmark exp.
# ./run.sh 8 4 64 0.99 8 0.0 True False False True False False False False bnmtest 3