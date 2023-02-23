#!/bin/bash
sbatch <<EOT
#!/bin/bash
# Example of running python script in a batch mode 
#SBATCH -c 1                       # Number of CPU Cores 
#SBATCH -p gpus                    # Partition (queue) 
#SBATCH --gres gpu:1               # gpu:n, where n = number of GPUs 
#SBATCH --mem 32G                  # memory pool for all cores 
#SBATCH --job-name='$1$2$9'
#SBATCH --exclude=lory
#SBATCH --output=slurm.%N.%j.log   # Standard output and error loga

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17}

# Source Virtual environment (conda)
. /vol/biomedic2/agk21/anaconda3/etc/profile.d/conda.sh
conda activate stylEx

./run.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17}

EOT