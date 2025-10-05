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

cd ../..

python3 src/run.py --train_dir dataset/Dataset_1_Cleaned --test_dir dataset/Dataset_2_Cleaned --output_dir output/L009 --seed 42 --model_name mobilenetv4_hybrid_medium.ix_e550_r384_in1k --no_amp --no_tqdm --epochs 200 >> L009.log 2>&1
