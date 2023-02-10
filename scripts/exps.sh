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

# ./slurm-wrapper.sh 8 8 64 0.99 9 0.0 False False False False True True True False testBM3 3
# ./slurm-wrapper.sh 8 8 64 0.99 9 0.0 False False False False True True True False testBM5 5

# ./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True False False True True True False testEigenEucCBR3 3
# ./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True True False True True True False testEigenCosineCBR3 3
# ./slurm-wrapper.sh 8 8 64 0.999 9 0.0 True True True False True True True False testEigenGumbleCBR3 3
# ./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True False False True True False False testEucCBR5 5
# ./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True True False True True False False testCosineCBR5 5
# ./slurm-wrapper.sh 8 8 64 0.999 9 0.0 True True True False True True False False testGumbleCBR5 5


./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True False False True True True False testEigenEucCBR3nok 3
./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True True False True True True False testEigenCosineCBR3nok 3
./slurm-wrapper.sh 8 8 64 0.999 9 0.0 True True True False True True True False testEigenGumbleCBR3nok 3
./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True False False True True False False testEucCBR5nok 5
./slurm-wrapper.sh 8 8 64 0.999 9 0.0 False True True False True True False False testCosineCBR5nok 5
./slurm-wrapper.sh 8 8 64 0.999 9 0.0 True True True False True True False False testGumbleCBR5nok 5
