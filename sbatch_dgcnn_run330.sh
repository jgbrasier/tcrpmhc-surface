#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-12:00
#SBATCH --mem=80G
#SBATCH -o out/DGCNN_run330_1layer12A_%j.out
#SBATCH -e out/DGCNN_run330_1layer12A_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbrasier@g.harvard.edu


. startup.sh

python -W ignore -u scripts/train_dmasif_run330.py --experiment_name DGCNN_1layer_12A_run330_nonegs_seed63 --batch_size 64 --embedding_layer DGCNN --random_rotation True --device cuda:0 --radius 12.0 --n_layers 1 --n_epoch 15 --seed 63

