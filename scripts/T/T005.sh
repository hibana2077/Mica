#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=12GB           
#PBS -l walltime=00:50:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4
source /scratch/rp06/sl5952/Mica/.venv/bin/activate
# mobilenetv3_large_100.miil_in21k_ft_in1k
cd ../..

python3 src/run.py --train_dir dataset/Dataset_1_Cleaned --test_dir dataset/Dataset_2_Cleaned --output_dir output/T005 --seed 42 --lr 4e-4 --model_name mobilenetv3_large_100.miil_in21k_ft_in1k --no_amp --no_tqdm --epochs 200 >> T005.log 2>&1