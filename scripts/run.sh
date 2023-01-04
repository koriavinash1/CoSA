ENCODERRES=$1
DECODERRES=$2
IMSIZE=$3
CBDECAY=$4
NCONCEPTS=$5
OPWEIGHTAGE=$6
QUANTIZE=$7
COSINE=$8
VAR=$9
BINARIZE=${10}
NAME=${11}


LOGS='/vol/biomedic2/agk21/PhDLogs/codes/ObjectDiscovery/testEigenSlots/LOGSJAN0423'
python /vol/biomedic2/agk21/PhDLogs/codes/ObjectDiscovery/testEigenSlots/train.py \
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
                                            --variational $VAR