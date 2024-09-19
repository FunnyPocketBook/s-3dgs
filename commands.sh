#!/bin/bash

python train.py -s /workspace/data/cup_plush -m /workspace/data/cup_plush_out_f3dgs -f sam -r 0 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 25000 30000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 25000 30000
python train.py -s /workspace/data/cup_plush -m /workspace/data/sibr_test -f sam -r 0 --speedup
python view.py -s /workspace/data/cup_plush/ -m /workspace/data/cup_plush_out_f3dgs --iteration 30000 -f sam
python render.py -s /workspace/data/cup_plush/ -m /workspace/data/cup_plush_out_f3dgs --iteration 30000 -f sam