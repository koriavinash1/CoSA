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

./slurm-wrapper.sh clevr hans3 False False False False False False test 3
./slurm-wrapper.sh clevr hans3 False False False False False False test 5
./slurm-wrapper.sh clevr hans7 False False False False False False test 5
./slurm-wrapper.sh clevr default False False False False False False test 5
./slurm-wrapper.sh bitmoji default False False False False False False test 5
./slurm-wrapper.sh multi_dsprites colored_on_colored False False False False False False test 5
./slurm-wrapper.sh multi_dsprites colored_on_grayscale False False False False False False test 5
./slurm-wrapper.sh objects_room default False False False False False False test 5
./slurm-wrapper.sh tetrominoes default False False False False False False test 5
./slurm-wrapper.sh ffhq default False False False False False False test 5
