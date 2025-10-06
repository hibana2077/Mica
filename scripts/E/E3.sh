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
# E3: Few-shot data efficiency (1,3,5 shots) comparing SSL vs MCR features
# Pre-train SSL
python3 src/ssl_mcr.py --train_dir dataset/Dataset_1_Cleaned --test_dir dataset/Dataset_2_Cleaned \
  --output_dir output/E3_ssl --model_name resnet18 --epochs 100 --batch_size 128 --mode ssl \
  --eval_geometry --geometry_k 10 --geometry_max_nodes 1200 --no_amp >> E3_ssl.log 2>&1
# Pre-train MCR
python3 src/ssl_mcr.py --train_dir dataset/Dataset_1_Cleaned --test_dir dataset/Dataset_2_Cleaned \
  --output_dir output/E3_mcr --model_name resnet18 --epochs 100 --batch_size 128 --mode mcr \
  --mcr_intra_k 10 --mcr_lambda_intra 1.0 --mcr_lambda_cross 1.0 --mcr_cross_k 5 \
  --mcr_intra_mode distance --eval_geometry --geometry_k 10 --geometry_max_nodes 1200 --no_amp >> E3_mcr.log 2>&1
# Few-shot eval on both checkpoints (k-NN)
python3 src/fewshot_eval.py --ckpt output/E3_ssl/ssl_model.pth --train_dir dataset/Dataset_1_Cleaned \
  --test_dir dataset/Dataset_2_Cleaned --shots 1 3 5 --method knn --repeats 10 >> E3_fewshot_ssl.log 2>&1
python3 src/fewshot_eval.py --ckpt output/E3_mcr/mcr_model.pth --train_dir dataset/Dataset_1_Cleaned \
  --test_dir dataset/Dataset_2_Cleaned --shots 1 3 5 --method knn --repeats 10 >> E3_fewshot_mcr.log 2>&1
