# Decomposition type ablation
# ENCODERRES=$1
# DECODERRES=$2
# IMSIZE=$3
# CBDECAY=$4
# NCONCEPTS=$5
# OPWEIGHTAGE=$6
# NOPOSITION=${7}
# QUANTIZE=$8
# COSINE=$9
# VAR=${10}
# CBQKEY=${11}
# CBRESTART=${12}
# EIGENQUANTIZER=${13}
# BINARIZE=${14}
# NAME=${15}
# ITER=${16}

./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True False False False False False False False benchmark1 1
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True False False False False False False False benchmark3 3
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True False False False False False False False benchmark5 5
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True False False False False False False False benchmark10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True False False False False False False False benchmark20 20


# ./slurm-wrapper.sh 8 8 64 0.75 16 0.1 True False False False False False False False benchmark3op 3

# mostly collapse
# ./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False False False False False euclidian_feature_quantize3 3
# ./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False True False False False euclidian_QKfeature_quantize3 3

# NaN errors
# ./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False False True False False euclidian_cbr_feature_quantize3 3
# ./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False True True False False euclidian_qk_cbr_feature_quantize3 3


./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False False False True True euclidian_eigen_quantize10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False False True True True euclidian_cbr_eigen_quantize10 10
./slurm-wrapper.sh 8 8 64 0.99 16 0.0 True True False False True False True True euclidian_qk_eigen_quantize3 3
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True False False True True True True euclidian_qk_cbr_eigen_quantize10 10

# ./slurm-wrapper.sh 8 8 64 0.75 16 0.1 True True False False True True True True euclidian_qk_cbr_eigen_quantize3op 3
# ./slurm-wrapper.sh 8 8 64 0.75 16 0.1 True True False True True True True True euclidian_qk_cbr_eigen_quantize3varop 3



./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False False False False False cosine_feature_quantize10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False True False False False cosine_qk_feature_quantize10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False False True False False cosine_cbr_feature_quantize10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False True True False False cosine_qk_cbr_feature_quantize10 10


./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False False False True True cosine_eigen_quantize10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False False True True True cosine_cbr_eigen_quantize10 10
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False True False True True cosine_qk_eigen_quantize3 3
./slurm-wrapper.sh 8 8 64 0.75 16 0.0 True True True False True True True True cosine_qk_cbr_eigen_quantize10 10


# ./slurm-wrapper.sh 8 8 64 0.75 16 0.1 True True True False True True True True cosine_qk_cbr_eigen_quantize3op 3
# ./slurm-wrapper.sh 8 8 64 0.75 16 0.1 True True True True True True True True cosine_qk_cbr_eigen_quantize3varop 3









