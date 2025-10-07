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

# k-NN evaluation using 100% labels from Dataset-1 as memory bank
python3 src/linear_probe.py \
  --encoder_ckpt output/SSL_resnet50/ssl_best.pth \
  --train_domain dataset/Dataset_1_Cleaned \
  --test_domain dataset/Dataset_2_Cleaned \
  --output_dir output/KNN_resnet50_D1toD2 \
  --label_fraction 1.0 --epochs 0 --batch_size 256 --lr 0.001 --no_amp >> S007.log 2>&1
