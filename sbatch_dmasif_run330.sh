#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-12:00
#SBATCH --mem=64G
#SBATCH -o out/dMaSIF_run330_3layer12A_%j.out
#SBATCH -e out/dMaSIF_run330_3layer12A_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbrasier@g.harvard.edu


. startup.sh

python -W ignore -u scripts/train_dmasif_run330.py --experiment_name dMaSIF_3layer_12A_run330_nonegs_seed3 --batch_size 64 --embedding_layer dMaSIF --random_rotation True --device cuda:0 --radius 12.0 --n_layers 3 --n_epoch 15 --seed 3 

