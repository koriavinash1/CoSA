# Decomposition type ablation
./slurm-wrapper.sh 4 4 64 0.0 8 25 False False benchmark
./slurm-wrapper.sh 4 4 64 0.0 8 25 True False quantize
./slurm-wrapper.sh 4 4 64 0.0 8 25 True True cosinequantize

./slurm-wrapper.sh 4 4 64 0.99 8 25 True False quantizeema
./slurm-wrapper.sh 4 4 64 0.99 8 25 True True cosinequantizeema