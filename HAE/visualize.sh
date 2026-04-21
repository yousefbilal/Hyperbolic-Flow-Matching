#!/bin/sh

#SBATCH --partition=gpu_a100  
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=vis_flowers
#SBATCH --time=24:00:00
#SBATCH --output=/home/fvaleau/HAE/logs/out-%x.%A.out
#SBATCH --error=/home/fvaleau/HAE/logs/err-%x.%A.err


module purge
module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0
module load GCCcore/13.3.0                  # GCC 12+ to match GLIBCXX


export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0"


export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Ensure GCC libraries are visible
export LD_LIBRARY_PATH=/sw/arch/RHEL9/gcc/13.3.0/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$(gcc --print-file-name=libstdc++.so.6)

export PYTHONPATH=/home/fvaleau/HAE:$PYTHONPATH
#----------------------------
# Clear old PyTorch extensions
#----------------------------
rm -rf ~/.cache/torch_extensions/


source $(conda info --base)/etc/profile.d/conda.sh
conda activate hyper_env

cd /gpfs/home6/fvaleau/HAE

#python Visualization/interpolation.py 
#python scripts/inference_umap.py
#python Visualization/decode_hyperemb.py
#python Visualization/decode_traj.py
#python FID/FID_calculation.py

#python Visualization/decode_hyperemb.py --dataset animals --emb_name fm_samples_animals_c0.5 --model_path animals_c0.5
#python Visualization/decode_hyperemb.py --dataset animals --emb_name fm_samples_animals_c1.0 --model_path /home/fvaleau/HAE/pretrained_models/hae_animalfaces.pt
#python Visualization/decode_hyperemb.py --dataset animals --emb_name fm_samples_animals_c2 --model_path animals_c2
#python Visualization/decode_hyperemb.py --dataset animals --emb_name fm_samples_animals_c3 --model_path animals_c3
#python Visualization/decode_hyperemb.py --dataset animals --emb_name fm_samples_animals_c5 --model_path animals_c5.0
#python Visualization/decode_hyperemb.py --dataset flowers --emb_name fm_samples_flowersc10 --model_path flowers_c10.0

python Visualization/decode_hyperemb.py --dataset animals --emb_name fm_samples_animals_euc --model_path /home/fvaleau/HAE/pretrained_models/hae_animalfaces.pt

