#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate feature_3dgs
TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.10
echo "Activated environment: $(conda info --envs | grep \* | awk '{print $1}')"
exec "$@"