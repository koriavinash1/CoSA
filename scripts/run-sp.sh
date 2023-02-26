DATASET=$1
VARIANT=$2
QUANTIZE=$3
COSINE=$4
GUMBLE=$5

LOGS='/vol/biomedic3/agk21/testEigenSlots2/LOGSBSA-Baseline'
TYPE='test'

if [ $QUANTIZE == 'True' ] && [ $COSINE == 'True' ] && [ $GUMBLE == 'False' ]
then
    TYPE='Cosine'
elif [ $QUANTIZE == 'True' ] && [ $COSINE == 'False' ] && [ $GUMBLE == 'False' ]
then
    TYPE='Euclidian'
elif [ $QUANTIZE == 'True' ] && [ $GUMBLE == 'True' ]
then
    TYPE='Gumble'
elif [ $QUANTIZE == 'False' ]
then
    TYPE='test'
    LOGS='/vol/biomedic3/agk21/testEigenSlots2/LOGSBMK'
fi


CONFIG=$LOGS'/ObjectDiscovery/'$DATASET$VARIANT$TYPE'/exp-parameters.json'
python /vol/biomedic3/agk21/testEigenSlots2/adhoc_setprediction.py \
                                            --learning_rate 0.001 \
                                            --config $CONFIG 