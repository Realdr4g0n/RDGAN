#!/bin/bash
#python3 train_fer.py --model Resnet101_0 --layer 0 --lr 0.0001 --dataroot FER2013
#python3 train_fer.py --model Resnet101_0 --layer 0 --lr 0.0001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet101_0 --layer 0 --lr 0.001 --dataroot FER2013
python3 train_fer.py --model Resnet101_0 --layer 0 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet101_0 --layer 0 --lr 0.01 --dataroot FER2013
python3 train_fer.py --model Resnet101_0 --layer 0 --lr 0.01 --dataroot FER_RD_16_BCE


python3 train_fer.py --model Resnet101_1 --layer 1 --lr 0.0001 --dataroot FER2013
python3 train_fer.py --model Resnet101_1 --layer 1 --lr 0.0001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet101_1 --layer 1 --lr 0.001 --dataroot FER2013
python3 train_fer.py --model Resnet101_1 --layer 1 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet101_1 --layer 1 --lr 0.01 --dataroot FER2013
python3 train_fer.py --model Resnet101_1 --layer 1 --lr 0.01 --dataroot FER_RD_16_BCE


python3 train_fer.py --model Resnet18_0 --layer 0 --lr 0.0001 --dataroot FER2013
python3 train_fer.py --model Resnet18_0 --layer 0 --lr 0.0001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet18_0 --layer 0 --lr 0.001 --dataroot FER2013
python3 train_fer.py --model Resnet18_0 --layer 0 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet18_0 --layer 0 --lr 0.01 --dataroot FER2013
python3 train_fer.py --model Resnet18_0 --layer 0 --lr 0.01 --dataroot FER_RD_16_BCE


python3 train_fer.py --model Resnet18_1 --layer 1 --lr 0.0001 --dataroot FER2013
python3 train_fer.py --model Resnet18_1 --layer 1 --lr 0.0001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet18_1 --layer 1 --lr 0.001 --dataroot FER2013
python3 train_fer.py --model Resnet18_1 --layer 1 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --model Resnet18_1 --layer 1 --lr 0.01 --dataroot FER2013
python3 train_fer.py --model Resnet18_1 --layer 1 --lr 0.01 --dataroot FER_RD_16_BCE
