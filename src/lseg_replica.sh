#!/bin/bash

cd encoders/lseg_encoder
#python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/room0/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/room0/images --workers 0
python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/room1/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/room1/images --workers 0
python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/office3/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/office3/images --workers 0
python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/office4/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/office4/images --workers 0
cd /workspace/feature_3dgs
python train.py -s /workspace/data/replica_data/room0 -m /workspace/output/replica_data/room0 -f lseg -r 0 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 25000 30000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 25000 30000
