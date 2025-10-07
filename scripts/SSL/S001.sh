#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=16GB           
#PBS -l walltime=06:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
source /scratch/rp06/sl5952/Mica/.venv/bin/activate

cd ../..

# VICReg + domain alignment pretraining (ResNet50)
python3 src/ssl_vicreg.py \
  --dataset1 dataset/Dataset_1_Cleaned \
  --dataset2 dataset/Dataset_2_Cleaned \
  --output_dir output/SSL_resnet50 \
  --model_name resnet50.a1_in1k \
  --epochs 200 --batch_size 128 --lr 0.002 --lambda_align 1.0 --proj_dim 2048 --proj_hidden 8192 --log_every 10 --no_amp >> S001.log 2>&1
