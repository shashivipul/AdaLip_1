#!/bin/bash

# Loop over the set of bounds
for bound in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  # Run the command with the current bound value
  python train.py --two_stage n --bound $bound --seed 45 --dataset cora --attack meta --ptb_rate 0.15 --epochs 200 --epochs_pre 400 --alpha 0 --gamma 1.0 --beta 0 --lr_optim 1e-2 --lr 1e-3
done

