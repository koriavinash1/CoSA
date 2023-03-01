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
TASKTYPE='OD'
# Baseline ------------------------
# ./slurm-wrapper.sh $TASKTYPE clevr default False False False False False False test 5
# ./slurm-wrapper.sh $TASKTYPE bitmoji default False False False False False False test 5
# ./slurm-wrapper.sh $TASKTYPE objects_room default False False False False False False test 5
# ./slurm-wrapper.sh $TASKTYPE tetrominoes default False False False False False False test 5
# ./slurm-wrapper.sh $TASKTYPE ffhq default False False False False False False test 5


# Euclidian codebook
./slurm-wrapper.sh $TASKTYPE clevr default True False False True False True Euclidian 5
./slurm-wrapper.sh $TASKTYPE bitmoji default True False False True False True Euclidian 5
./slurm-wrapper.sh $TASKTYPE objects_room default True False False True False True Euclidian 5
./slurm-wrapper.sh $TASKTYPE tetrominoes default True False False True False True Euclidian 5
./slurm-wrapper.sh $TASKTYPE ffhq default True False False True False True Euclidian 5



# Cosine codebook
./slurm-wrapper.sh $TASKTYPE clevr default True True False True False True Cosine 5
./slurm-wrapper.sh $TASKTYPE bitmoji default True True False True False True Cosine 5
./slurm-wrapper.sh $TASKTYPE objects_room default True True False True False True Cosine 5
./slurm-wrapper.sh $TASKTYPE tetrominoes default True True False True False True Cosine 5
./slurm-wrapper.sh $TASKTYPE ffhq default True True False True False True Cosine 5


# Gumble codebook
./slurm-wrapper.sh $TASKTYPE clevr default True False True True False True Gumble 5
./slurm-wrapper.sh $TASKTYPE bitmoji default True False True True False True Gumble 5
./slurm-wrapper.sh $TASKTYPE objects_room default True False True True False True Gumble 5
./slurm-wrapper.sh $TASKTYPE tetrominoes default True False True True False True Gumble 5
./slurm-wrapper.sh $TASKTYPE ffhq default True False True True False True Gumble 5


# =========================================================================
TASKTYPE='SP'
# Baseline ------------------------
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 False False False
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 False False False
# ./slurm-wrapper.sh $TASKTYPE ffhq default False False False

# Euclidian codebook
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 True False False
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 True False False
# ./slurm-wrapper.sh $TASKTYPE ffhq default True False False

# Cosine codebook
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 True True False
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 True True False
# ./slurm-wrapper.sh $TASKTYPE ffhq default True True False

# Gumble codebook
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 True False True
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 True False True
# ./slurm-wrapper.sh $TASKTYPE ffhq default True False True



# =========================================================================
TASKTYPE='RE'

#  Baseline ------------------------
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 False False False default
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 False False False default
# ./slurm-wrapper.sh $TASKTYPE ffhq default False False False default

# Euclidian codebook
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 True False False default
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 True False False default
# ./slurm-wrapper.sh $TASKTYPE ffhq default True False False default

# Cosine codebook
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 True True False default
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 True True False default
# ./slurm-wrapper.sh $TASKTYPE ffhq default True True False default

# Gumble codebook
# ./slurm-wrapper.sh $TASKTYPE clevr hans3 True False True default
# ./slurm-wrapper.sh $TASKTYPE clevr hans7 True False True default
# ./slurm-wrapper.sh $TASKTYPE ffhq default True False True default
