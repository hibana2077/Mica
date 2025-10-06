#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=16GB           
#PBS -l walltime=01:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
source /scratch/rp06/sl5952/Mica/.venv/bin/activate

cd ../..
# E2: Cross-source alignment metrics (neighborhood overlap etc.)
# Ensure geometry evaluation enabled to capture cross_source_overlap
python3 src/ssl_mcr.py --train_dir dataset/Dataset_1_Cleaned \
  --test_dir dataset/Dataset_2_Cleaned --output_dir output/E2_mcr --model_name resnet18 \
  --epochs 80 --batch_size 128 --mode mcr --mcr_intra_k 10 --mcr_lambda_intra 1.0 \
  --mcr_lambda_cross 1.0 --mcr_cross_k 5 --mcr_intra_mode distance --eval_geometry \
  --geometry_k 15 --geometry_max_nodes 1500 --no_amp >> E2_mcr.log 2>&1
