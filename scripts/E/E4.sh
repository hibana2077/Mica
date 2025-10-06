#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=16GB           
#PBS -l walltime=04:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
source /scratch/rp06/sl5952/Mica/.venv/bin/activate

cd ../..
# E4: Ablations over lambda weights, intra modes, and k
# Define arrays
LAM_INTRA=(0.5 1.0 2.0)
LAM_CROSS=(0.5 1.0 2.0)
INTRA_MODE=(distance laplacian)
K_VALUES=(5 10 15)
EPOCHS=60
BATCH=128

for li in "${LAM_INTRA[@]}"; do
  for lc in "${LAM_CROSS[@]}"; do
    for im in "${INTRA_MODE[@]}"; do
      for k in "${K_VALUES[@]}"; do
        OUT=output/E4_li${li}_lc${lc}_im${im}_k${k}
        LOG=E4_li${li}_lc${lc}_im${im}_k${k}.log
        echo "Running ablation li=$li lc=$lc im=$im k=$k" >> E4_master.log
        python3 src/ssl_mcr.py --train_dir dataset/Dataset_1_Cleaned --test_dir dataset/Dataset_2_Cleaned \
          --output_dir ${OUT} --model_name resnet18 --epochs ${EPOCHS} --batch_size ${BATCH} --mode mcr \
          --mcr_intra_k ${k} --mcr_lambda_intra ${li} --mcr_lambda_cross ${lc} --mcr_cross_k 5 \
          --mcr_intra_mode ${im} --eval_geometry --geometry_k 10 --geometry_max_nodes 1000 --no_amp >> ${LOG} 2>&1
      done
    done
  done
done
