#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaV100:1
#SBATCH -t 0-12:00
#SBATCH --mem=64G
#SBATCH -o out/dMaSIF_run329_3layer12A.out
#SBATCH -e out/dMaSIF_run329_3layer12A.err
#SBATCH --mail-user=jbrasier@g.harvard.edu

module load conda 
module load gcc/9.2.0 cuda/11.7

conda activate dmasif

python -W ignore -u scripts/train_dmasif_run329.py --experiment_name dMaSIF_search_3layer_12A --batch_size 64 --embedding_layer dMaSIF --search True --device cuda:0 --random_rotation True --radius 12.0 --n_layers 3
