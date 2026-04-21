#!/bin/sh

#SBATCH --partition=gpu_h100  
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=inference_flowers
#SBATCH --time=4:00:00
#SBATCH --output=/home/fvaleau/HAE/logs/out-%x.%A.out
#SBATCH --error=/home/fvaleau/HAE/logs/err-%x.%A.err
#SBATCH --mem=64G


module purge
module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0
module load GCCcore/13.3.0                  # GCC 12+ to match GLIBCXX



export WANDB_API_KEY="367c7b421cacfce69bdca1a2e951a0568341d348"


echo "=== DEBUG INFO ==="
which python
python --version
python -c "import sys; print(sys.prefix)"
python -c "import torch; print('torch:', torch.__version__)" || echo "torch not found"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)" || echo "matplotlib not found"
echo "=================="



export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0"


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


#python scripts/train.py \
    # --dataset_type=flowers_encode_eva \
    # --psp_checkpoint_path=pretrained_models/psp_flowers.pt \
    # --exp_dir=output_gpu \
    # --feature_size=512 \
    # --workers=8 \
    # --batch_size=8 \
    # --test_batch_size=8 \
    # --test_workers=8 \
    # --val_interval=80000 \
    # --save_interval=5000 \
    # --encoder_type=GradualStyleEncoder \
    # --start_from_latent_avg \
    # --lpips_lambda=1 \
    # --l2_lambda=1 \
    # --image_interval=1000 \
    # --hyperbolic_lambda=0.3 \
    # --reverse_lambda=1 

# python scripts/inference.py \
# --exp_dir=inference_out \
# --checkpoint_path=pretrained_models/hae_flowers.pt \
# --data_path=proj/flowers/test \
# --test_batch_size=4 \
# --test_workers=4

# THIS IS THE RIGHT JOB SCRIPT, NOT VISUALIZE
#--checkpoint_path /home/fvaleau/HAE/pretrained_models/hae_flowers.pt \

# python scripts/inference_umap.py \
#     --exp_dir proj/flowers_emb_10 \
#     --checkpoint_path /scratch-shared/fvaleau/output_flowers_c10.0/checkpoints/best_model.pt \
#     --data_path /home/fvaleau/HAE/proj/flowers/train \
#     --test_batch_size 1

# python scripts/inference_umap.py \
#     --exp_dir proj/animals_emb_0.5 \
#     --checkpoint_path /scratch-shared/fvaleau/output_animals_c0.5/checkpoints/best_model.pt \
#     --data_path /home/fvaleau/HAE/proj/animal_faces \
#     --test_batch_size 1

# python scripts/inference_umap.py \
#     --exp_dir proj/animals_emb_1 \
#     --checkpoint_path /scratch-shared/fvaleau/output_animals_c1/checkpoints/best_model.pt \
#     --data_path /home/fvaleau/HAE/proj/animal_faces \
#     --test_batch_size 1

# python scripts/inference_umap.py \
#     --exp_dir proj/animals_emb_2 \
#     --checkpoint_path /scratch-shared/fvaleau/output_animals_c2/checkpoints/best_model.pt \
#     --data_path /home/fvaleau/HAE/proj/animal_faces \
#     --test_batch_size 1

# python scripts/inference_umap.py \
#     --exp_dir proj/animals_emb_3 \
#     --checkpoint_path /scratch-shared/fvaleau/output_animals_c3/checkpoints/best_model.pt \
#     --data_path /home/fvaleau/HAE/proj/animal_faces \
#     --test_batch_size 1

python scripts/inference_umap.py \
    --exp_dir proj/animals_emb_5 \
    --checkpoint_path /scratch-shared/fvaleau/output_animals_c5.0/checkpoints/best_model.pt \
    --data_path /home/fvaleau/HAE/proj/animal_faces \
    --test_batch_size 1

# python scripts/inference_umap.py \
#     --exp_dir proj/animals_emb_10.0 \
#     --checkpoint_path /scratch-shared/fvaleau/output_animals_c5.0/checkpoints/best_model.pt \
#     --data_path /home/fvaleau/HAE/proj/animal_faces \
#     --test_batch_size 1