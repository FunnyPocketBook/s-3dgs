#!/bin/bash
# cd /workspace/feature-3dgs/encoders/lseg_encoder
# python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/office3/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/office3/images --workers 0
# python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/office4/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/office4/images --workers 0
# python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/replica_data/room1/rgb_feature_langseg --test-rgb-dir /workspace/data/replica_data/room1/images --workers 0
# cd /workspace/feature-3dgs/encoders/sam_encoder
# python segment.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --data /workspace/output/tree18_sam/ --iteration 7000
# python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input /workspace/data/replica_data/office3/images  --output /workspace/data/replica_data/office3/sam_embeddings
# python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input /workspace/data/replica_data/office4/images  --output /workspace/data/replica_data/office4/sam_embeddings
# python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input /workspace/data/replica_data/room0/images  --output /workspace/data/replica_data/room0/sam_embeddings
# python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input /workspace/data/cup_plush/images  --output /workspace/data/cup_plush/sam_embeddings
# python -u encode_images.py --backbone clip_vitl16_384 --weights /workspace/lseg/demo_e200.ckpt --widehead --no-scaleinv --outdir /workspace/data/tree18/colmap/rgb_feature_langseg --test-rgb-dir /workspace/data/tree18/colmap/images --workers 0
cd /workspace/feature-3dgs/
python train.py -s /workspace/data/tree/colmap -m /workspace/output/tree/lseg/no_speedup/home/r-1 -f lseg --save_iterations 100 200 300 400 500 590 2000 5000 7000 --checkpoint_iterations 100 200 300 400 500 590 2000 5000 7000
# python render.py -s /workspace/data/tree18/colmap -m /workspace/output/tree18_sam/ -f sam --iteration 7000 --novel_view --multi_interpolate --video 
# python render.py -s /workspace/data/tree18/colmap -m /workspace/output/tree18_lseg/ -f lseg --iteration 20000 --novel_view --multi_interpolate --video
# python train.py -s /workspace/data/tree18/colmap -m /workspace/output/tree18/lseg/ -f lseg -r 4 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --eval --iterations 20000
# python train.py -s /workspace/data/replica_data/room0 -m /workspace/output/room0/sam -f sam -r 2 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --eval --iterations 20000 --test_iterations 5000 7000 15000 20000
# python train.py -s /workspace/data/replica_data/room1 -m /workspace/output/room1/sam -f sam -r 2 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --eval --iterations 20000 --test_iterations 5000 7000 15000 20000
# python train.py -s /workspace/data/replica_data/office3 -m /workspace/output/office3/sam -f sam -r 2 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --eval --iterations 20000 --test_iterations 5000 7000 15000 20000
# python train.py -s /workspace/data/replica_data/office4 -m /workspace/output/office4/sam -f sam -r 2 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --eval --iterations 20000 --test_iterations 5000 7000 15000 20000
# python train.py -s /workspace/data/cup_plush -m /workspace/output/cup_plush_sam -f sam -r 4 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 --eval --iterations 20000 --test_iterations 5000 7000 15000 20000 --start_checkpoint /workspace/output/cup_plush_sam/chkpnt5000.pth
# python train.py -s /workspace/data/cup_plush -m /workspace/output/feature_3dgs/cup_plush_sam -f sam -r 0 --speedup --save_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 25000 30000 --checkpoint_iterations 100 200 300 400 500 600 700 800 900 1000 2000 5000 7000 10000 15000 20000 25000 30000 --eval
# python view.py -s /workspace/data/cup_plush -m /workspace/output/cup_plush_sam -f sam --iteration 20000
# python view.py -s /workspace/data/tree18 -m /workspace/output/tree18_lseg -f lseg --iteration 7000
# python view2.py -s /workspace/data/cup_plush -m /workspace/output/cup_plush_mcmc --iteration 30000
# python render.py -s /workspace/data/cup_plush -m /workspace/output/cup_plush_sam -f sam --iteration 20000 --novel_view --video --multi_interpolate
# python render.py -s /workspace/data/replica_data/room0 -m /workspace/output/room0/sam -f sam --iteration 20000 --novel_view --video --multi_interpolate
# python render.py -s /workspace/data/replica_data/room1 -m /workspace/output/room1/sam -f sam --iteration 20000 --novel_view --video --multi_interpolate
# python render.py -s /workspace/data/replica_data/office3 -m /workspace/output/office3/sam -f sam --iteration 20000 --novel_view --video --multi_interpolate
# python render.py -s /workspace/data/replica_data/office4 -m /workspace/output/office4/sam -f sam --iteration 20000 --novel_view --video --multi_interpolate
# python render.py -s /workspace/data/tree18/colmap -m /workspace/output/tree18_sam -f sam --iteration 20000 --novel_view --video --multi_interpolate
# cd /workspace/feature-3dgs/encoders/lseg_encoder
# python -u segmentation.py --data /workspace/output/cup_plush_lseg --weights /workspace/lseg/demo_e200.ckpt --iteration 2000
