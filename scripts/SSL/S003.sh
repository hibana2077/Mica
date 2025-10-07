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

# Linear probe 5% labels (Dataset-1 -> Dataset-2)
python3 src/linear_probe.py \
  --encoder_ckpt output/SSL_resnet50/ssl_best.pth \
  --train_domain dataset/Dataset_1_Cleaned \
  --test_domain dataset/Dataset_2_Cleaned \
  --output_dir output/LP_resnet50_D1toD2_f05 \
  --label_fraction 0.05 --epochs 100 --batch_size 128 --lr 0.001 --no_amp >> S003.log 2>&1
