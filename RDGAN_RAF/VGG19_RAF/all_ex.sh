#!/bin/bash

printf "16 , 0 , lr 0.001"
python3 train_raf.py --model VGG16 --layer 0 --lr 0.001 --dataroot RAF_single_RDGAN 

printf "16 , 0 , lr 0.01"
python3 train_raf.py --model VGG16 --layer 0 --lr 0.01 --dataroot RAF_single_RDGAN

printf "16 , 1 , lr 0.001"
python3 train_raf.py --model VGG16 --layer 1 --lr 0.001 --dataroot RAF_single_RDGAN

printf "16 , 1 , lr 0.01"
python3 train_raf.py --model VGG16 --layer 1 --lr 0.01 --dataroot RAF_single_RDGAN

printf "16 , 2 , lr 0.001"
python3 train_raf.py --model VGG16 --layer 2 --lr 0.001 --dataroot RAF_single_RDGAN

printf "16 , 2 , lr 0.01"
python3 train_raf.py --model VGG16 --layer 2 --lr 0.01 --dataroot RAF_single_RDGAN


printf "19 , 0 , lr 0.001"
python3 train_raf.py --model VGG19 --layer 0 --lr 0.001 --dataroot RAF_single_RDGAN

printf "19 , 0 , lr 0.01"
python3 train_raf.py --model VGG19 --layer 0 --lr 0.01 --dataroot RAF_single_RDGAN

printf "19 , 1 , lr 0.001"
python3 train_raf.py --model VGG19 --layer 1 --lr 0.001 --dataroot RAF_single_RDGAN

printf "19 , 1 , lr 0.01"
python3 train_raf.py --model VGG19 --layer 1 --lr 0.01 --dataroot RAF_single_RDGAN

printf "19 , 2 , lr 0.001"
python3 train_raf.py --model VGG19 --layer 2 --lr 0.001 --dataroot RAF_single_RDGAN

printf "19 , 2 , lr 0.01"
python3 train_raf.py --model VGG19 --layer 2 --lr 0.01 --dataroot RAF_single_RDGAN
