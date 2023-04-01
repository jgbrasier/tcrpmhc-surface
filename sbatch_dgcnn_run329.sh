#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaV100:1
#SBATCH -t 0-12:00
#SBATCH --mem=64G
#SBATCH -o out/DGCNN_run329_3layer12A_%j.out
#SBATCH -e out/DGCNN_run329_3layer12A_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbrasier@g.harvard.edu

. startup.sh

python -W ignore -u scripts/train_dmasif_run329.py --experiment_name DGCNN_search_3layer_12A --batch_size 64 --embedding_layer DGCNN --search True --device cuda:0 --random_rotation True --radius 12.0 --n_layers 3 --n_epoch 20
