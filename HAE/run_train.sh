#!/bin/sh

#SBATCH --partition=gpu_h100  
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=training_animals
#SBATCH --time=50:00:00
#SBATCH --output=/home/fvaleau/HAE/logs/out-%x.%A.out
#SBATCH --error=/home/fvaleau/HAE/logs/err-%x.%A.err


#module load cuda/12.2  # or whatever version is available
module purge
# module load 2023
# module load Miniconda3/23.5.2-0 
# module load CUDA/12.1.1 
# module load GCCcore/12.3.0
# source activate hae_env

# python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
# python -c "import torch; import matplotlib; print(torch.version.cuda); print(torch.cuda.is_available())"

module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0
module load GCCcore/13.3.0                  # GCC 12+ to match GLIBCXX



export WANDB_API_KEY="367c7b421cacfce69bdca1a2e951a0568341d348"

#conda env create -f environment/hae_env.yaml -n hae_env_2023
#source activate hae_env_2023
#source $(conda info --base)/etc/profile.d/conda.sh
#conda activate hae_env_2023

#source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh
#conda activate /home/jrosenthal/.conda/envs/hae_env_2023


echo "=== DEBUG INFO ==="
which python
python --version
python -c "import sys; print(sys.prefix)"
python -c "import torch; print('torch:', torch.__version__)" || echo "torch not found"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)" || echo "matplotlib not found"
echo "=================="



TORCH_CUDA_ARCH_LIST="8.0;9.0"  # or set to all supported architectures
export TORCH_CUDA_ARCH_LIST

export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Ensure GCC libraries are visible
export LD_LIBRARY_PATH=/sw/arch/RHEL9/gcc/13.3.0/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$(gcc --print-file-name=libstdc++.so.6)


#----------------------------
# Clear old PyTorch extensions
#----------------------------
rm -rf ~/.cache/torch_extensions/


source $(conda info --base)/etc/profile.d/conda.sh
conda activate hyper_env

cd /gpfs/home6/fvaleau/HAE

# python scripts/train.py \
#     --dataset_type=flowers_encode_eva \
#     --psp_checkpoint_path=pretrained_models/psp_flowers.pt \
#     --exp_dir=/scratch-shared/fvaleau/output_flowers_c10.0 \
#     --feature_size=512 \
#     --workers=8 \
#     --batch_size=8 \
#     --test_batch_size=8 \
#     --test_workers=8 \
#     --val_interval=80000 \
#     --save_interval=10000 \
#     --encoder_type=GradualStyleEncoder \
#     --start_from_latent_avg \
#     --lpips_lambda=1 \
#     --l2_lambda=1 \
#     --image_interval=1000 \
#     --hyperbolic_lambda=0.3 \
#     --reverse_lambda=1 \
#     --hyperbolic_curvature=-10 \
#     --use_wandb

python scripts/train.py \
    --dataset_type=animalfaces_encode_eva \
    --psp_checkpoint_path=pretrained_models/psp_animalfaces.pt \
    --exp_dir=/scratch-shared/fvaleau/output_animals_c10 \
    --feature_size=512 \
    --workers=8 \
    --batch_size=8 \
    --test_batch_size=8 \
    --test_workers=8 \
    --val_interval=80000 \
    --save_interval=10000 \
    --encoder_type=GradualStyleEncoder \
    --start_from_latent_avg \
    --lpips_lambda=1 \
    --l2_lambda=1 \
    --image_interval=1000 \
    --hyperbolic_lambda=0.3 \
    --reverse_lambda=1 \
    --hyperbolic_curvature=-10 \
    --use_wandb


