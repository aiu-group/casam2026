#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

ARCH=resnet56_cifar10
SEED=2

python train.py --data_name cifar10 --num_classes 10 --arch $ARCH --optimizer sgd --momentum 0.9 --learning_rate 0.1 --batch_size_train 128 --weight_decay 1.0e-4 --gpu_id 0 --epochs 200 --scheduler cosine --seed $SEED --sam_rho 0.05 --use_casam 1 --casam_init_gamma 0.0 --casam_end_gamma -2.0 --wandb 1 --corrupt_prob 0.0 --adaptive 0 --squeeze arctan

python train.py --data_name cifar10 --num_classes 10 --arch $ARCH --optimizer sgd --momentum 0.9 --learning_rate 0.1 --batch_size_train 128 --weight_decay 1.0e-4 --gpu_id 0 --epochs 200 --scheduler cosine --seed $SEED --sam_rho 0.2 --use_casam 1 --casam_init_gamma 0.0 --casam_end_gamma -2.0 --wandb 1 --corrupt_prob 0.3 --adaptive 0 --squeeze arctan
