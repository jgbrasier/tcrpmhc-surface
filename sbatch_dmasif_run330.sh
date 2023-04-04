#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaV100:1
#SBATCH -t 0-12:00
#SBATCH --mem=64G
#SBATCH -o out/dMaSIF_run330_3layer12A_%j.out
#SBATCH -e out/dMaSIF_run330_3layer12A_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbrasier@g.harvard.edu


. startup.sh

python -W ignore -u scripts/train_dmasif_run330.py --experiment_name dMaSIF_search_3layer_12A_nonegs --batch_size 64 --embedding_layer dMaSIF --search True --device cuda:0 --random_rotation True --radius 12.0 --n_layers 3 --n_epoch 20
