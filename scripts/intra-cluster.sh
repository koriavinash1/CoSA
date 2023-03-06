# DATASET=$1
# VARIANT=$2
# QUANTIZE=$3
# COSINE=$4
# GUMBLE=$5
# CBQKEY=$6
# CBRESTART=$7
# EIGENQUANTIZER=$8
# NAME=$9
# ITER=${10}

# =============================================================
TASKTYPE='RE'

./run-re.sh clevr hans3 False False False Default Baseline &
./run-re.sh clevr hans3 True False False Default Euclidian &
./run-re.sh clevr hans3 True True False Default Cosine &
./run-re.sh clevr hans3 True False True Default Gumble 