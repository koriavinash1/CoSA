DATASET=$1
VARIANT=$2
REASONINGTYPE=$3


# QUANTIZE=$3
# COSINE=$4
# GUMBLE=$5
# NAME=$7

LOGS='/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI'

echo 'RUNNING all REASONING tasks models saved in'$LOGS


NAME='DefaultCNN'
python /vol/biomedic3/agk21/testEigenSlots2/baselineclassifier.py \
                                            --dataset_name $DATASET \
                                            --variant $VARIANT \
                                            --exp_name $NAME \
                                            --reasoning_type $REASONINGTYPE \
                                            --model_dir $LOGS'/'$DATASET$VARIANT$REASONINGTYPE  &



NAME='Baseline'
python /vol/biomedic3/agk21/testEigenSlots2/reasoning.py \
                                            --dataset_name $DATASET \
                                            --variant $VARIANT \
                                            --quantize False \
                                            --cosine False \
                                            --gumble False \
                                            --reasoning_type $REASONINGTYPE \
                                            --learning_rate 0.0004 \
                                            --exp_name $NAME \
                                            --model_dir $LOGS'/'$DATASET$VARIANT$REASONINGTYPE &


NAME='Euclidian'
python /vol/biomedic3/agk21/testEigenSlots2/reasoning.py \
                                            --dataset_name $DATASET \
                                            --variant $VARIANT \
                                            --quantize True \
                                            --cosine False \
                                            --gumble False \
                                            --reasoning_type $REASONINGTYPE \
                                            --learning_rate 0.0004 \
                                            --exp_name $NAME \
                                            --model_dir $LOGS'/'$DATASET$VARIANT$REASONINGTYPE &


NAME='Cosine'
python /vol/biomedic3/agk21/testEigenSlots2/reasoning.py \
                                            --dataset_name $DATASET \
                                            --variant $VARIANT \
                                            --quantize True \
                                            --cosine True \
                                            --gumble False \
                                            --reasoning_type $REASONINGTYPE \
                                            --learning_rate 0.0004 \
                                            --exp_name $NAME \
                                            --model_dir $LOGS'/'$DATASET$VARIANT$REASONINGTYPE &


NAME='Gumble'
python /vol/biomedic3/agk21/testEigenSlots2/reasoning.py \
                                            --dataset_name $DATASET \
                                            --variant $VARIANT \
                                            --quantize True \
                                            --cosine False \
                                            --gumble True \
                                            --reasoning_type $REASONINGTYPE \
                                            --learning_rate 0.0004 \
                                            --exp_name $NAME \
                                            --model_dir $LOGS'/'$DATASET$VARIANT$REASONINGTYPE &



python /vol/biomedic3/agk21/dummy.py