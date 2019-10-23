#!/bin/bash
python3 train.py --dataroot ./datasets/RAF_D2N/ --model cycle_gan --name RDGAN_RAF_1 --load_size 100 --crop_size 100 --batch_size 1 --niter 150 --niter_decay 150 --continue_train --epoch_count 151

