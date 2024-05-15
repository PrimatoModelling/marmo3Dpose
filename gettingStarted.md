# Getting started 
This document explains how to run the demo.

# Additional dataset for the demo
Please download the video data and pretrained models from [DOI:10.5281/zenodo.11174658](https://zenodo.org/records/11174658).   
Place the unzipped contents ("videos" and "weight" folders) in the same directory as this file. 

# GPU driver 
    # Linux: >= 450.80.02
    # Windows: >=452.39
    # Tested 
    #   CentOS 7, NVIDIA UNIX x86_64 Kernel Module  470.57.02 
    #   Ubuntu 20.04, NVIDIA UNIX x86_64 Kernel Module  510.47.03 

# Python version 
3.10.12 

# Installing tools
```bash
cd $PathForDownloadedFolder

# pytorch 
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Openmmlab
pip install openmim==0.3.9
mim install mmcv-full==1.6.2
mim install mmpose==0.29.0
mim install mmdet==2.26.0
mim install mmtrack==0.14.0
mim install mmcls==0.25.0
pip install xtcocotools==1.12 # needed to be downgraded due to compatibility to numpy

# Major tools
pip install pip install opencv-contrib-python==4.8.1.78
pip install numba==0.58.0
pip install h5py==3.9.0
pip install pyyaml==6.0.1
pip install toml==0.10.2 
pip install matplotlib==3.8.0
pip install joblib==1.3.2

# Minor or local resources 
pip install imgstore==0.2.9
pip install 'src/m_lib'

# Others installed with mmcv-full 
# tqdm==4.65.0
# scipy==1.7.3
# cv2==4.8.0
```

# Run demo 
```bash 
bash run_demo.sh # the results will appear at "./results"
```
