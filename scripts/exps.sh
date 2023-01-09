# Decomposition type ablation
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 False False False False False benchmark3 3
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 False False False False False benchmark5 5
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 False False False False False benchmark10 10


./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False False True euclidian_quantization_noposition3 3
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False True True euclidian_quantization_cov_binary_noposition3 3

./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False False False euclidian_quantization3 3
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False True False euclidian_quantization_cov_binary3 3

./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False False False cosine_quantization3 3
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False True False cosine_quantization_cov_binary3 3

./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False False True cosine_quantization_noposition3 3
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False True True cosine_quantization_cov_binary_noposition3 3

./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False False True cosine_quantization_noposition5 5
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False True True cosine_quantization_cov_binary_noposition5 5
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False False True euclidian_quantization_noposition5 5
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False True True euclidian_quantization_cov_binary_noposition5 5

./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False False True cosine_quantization_noposition10 10
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True True False True True cosine_quantization_cov_binary_noposition10 10
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False False True euclidian_quantization_noposition10 10
./slurm-wrapper.sh 4 4 64 0.99 16 0.0 True False False True True euclidian_quantization_cov_binary_noposition10 10


./slurm-wrapper.sh 8 8 128 0.99 16 0.0 True False False True True HRES_euclidian_quantization_cov_binary_noposition3 3
./slurm-wrapper.sh 8 8 128 0.99 16 0.0 True True True True True HRES_cosine_quantization_cov_binary_variational_noposition3 3

./slurm-wrapper.sh 8 8 128 0.99 16 0.0 True False False True True HRES_euclidian_quantization_cov_binary_noposition5 5
./slurm-wrapper.sh 8 8 128 0.99 16 0.0 True True True True True HRES_cosine_quantization_cov_binary_variational_noposition5 5

./slurm-wrapper.sh 8 8 128 0.99 16 0.0 True False False True True HRES_euclidian_quantization_cov_binary_noposition10 10
./slurm-wrapper.sh 8 8 128 0.99 16 0.0 True True True True True HRES_cosine_quantization_cov_binary_variational_noposition10 10


# ./slurm-wrapper.sh 8 8 128 0.0 16 0.99 True False True True True HRES_euclidian_quantization_cov_binary_variational_noposition 3
# ./slurm-wrapper.sh 8 8 128 0.0 16 0.99 True True False True True HRES_cosine_quantization_cov_binary_noposition 3

# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True False True False False euclidian_quantization_variational 3
# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True False True False True euclidian_quantization_variational_noposition 3
# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True False True True False euclidian_quantization_cov_binary_variational 3
# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True False True True True euclidian_quantization_cov_binary_variational_noposition 3


# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True True True False False cosine_quantization_variational 3
# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True True True False True cosine_quantization_variational_noposition 3
# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True True True True False cosine_quantization_cov_binary_variational 3
# ./slurm-wrapper.sh 4 4 64 0.0 16 0.99 True True True True True cosine_quantization_cov_binary_variational_noposition 3

