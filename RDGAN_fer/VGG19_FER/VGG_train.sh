#!/bin/bash
python3 train_fer.py --layer 2 --lr 0.001 --dataroot FER2013
python3 train_fer.py --layer 2 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --layer 2 --lr 0.01 --dataroot FER2013
python3 train_fer.py --layer 2 --lr 0.01 --dataroot FER_RD_16_BCE

python3 train_fer.py --layer 1 --lr 0.001 --dataroot FER2013
python3 train_fer.py --layer 1 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --layer 1 --lr 0.01 --dataroot FER2013
python3 train_fer.py --layer 1 --lr 0.01 --dataroot FER_RD_16_BCE

python3 train_fer.py --layer 0 --lr 0.001 --dataroot FER2013
python3 train_fer.py --layer 0 --lr 0.001 --dataroot FER_RD_16_BCE

python3 train_fer.py --layer 0 --lr 0.01 --dataroot FER2013
python3 train_fer.py --layer 0 --lr 0.01 --dataroot FER_RD_16_BCE

 
