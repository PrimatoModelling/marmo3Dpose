#!/bin/bash 

# dataset name 
data_name='dl_pogz4_ctrl_cj634cj450_cj886_20221125_120000'

# path for video data
raw_data_dir='./videos'

# output directories
svid2dout_dir='./results'
vid3dout_dir='./results'
label2d_dir='./results'
results3d_dir='./results'

# gpu device
device_str='cuda:0'

# reference camera for time synchronization
camName=23506239

# frame range for analysis
skipFrame=0
procFrame=3000

# 2D analysis parameters
config_path='./calib/marmo/config.yaml'
tracking_config='model/track/tk_bytetrack_demo.py'
pose_config='model/pose/tk_hrnet_w32_256x256_V0p3.py'
pose_checkpoint='weight/pose.pth'
id_config='model/id/tk_resnet50_8xb32_RBI_v4.py'
id_checkpoint='weight/ID.pth'

# 3D analysis parameters
config_3d_toml='./config_tmpl.toml'
calib_3d_toml='./calibration_tmpl.toml'
fps=24
n_kp=18
thr_kp_detection=0.5

## Run 2D analysis
#
python ./process_2d.py \
       --config_path ${config_path} \
        --data_name ${data_name} \
        --raw_data_dir ${raw_data_dir} \
        --label2d_dir ${label2d_dir} \
        --device_str ${device_str} \
        --tracking_config ${tracking_config} \
        --pose_config ${pose_config} \
        --pose_checkpoint ${pose_checkpoint} \
        --id_config ${id_config} \
        --id_checkpoint ${id_checkpoint} \
        --procFrame ${procFrame}

## make video for results of 2D analysis
#
n_frame_to_save=500
label2d_output_dir=${label2d_dir}
python ./visualize_2D.py  \
        --path_vid ${raw_data_dir}/${data_name}.${camName}/000000.mp4  \
        --path_json ${label2d_output_dir}/${data_name}/${data_name}_${camName}_000000.json \
        --path_output ${vid2dout_dir}/${data_name}_${camName}_000000.mp4 \
        --n_frame_to_save ${n_frame_to_save}


## Run 3D analysis
#
tmp1=$(( $skipFrame/$fps ))
tmp2=$(( $procFrame+$skipFrame ))
tmp2=$(( $tmp2/$fps ))
t_intv="${tmp1},${tmp2}"
echo $t_intv

# t_intv="None"
python ./process_3d_multi.py \
        --config_3d_toml ${config_3d_toml}\
        --calib_3d_toml ${calib_3d_toml}\
        --config_path ${config_path}\
        --fps ${fps}\
        --t_intv ${t_intv}\
        --n_kp ${n_kp} \
        --thr_kp_detection ${thr_kp_detection}\
        --results3d_dir ${results3d_dir} \
        --raw_data_dir ${raw_data_dir}\
        --label2d_dir ${label2d_dir}\
        --data_name ${data_name}  

## Make video for results of 3D 
#
pickledata_dir=${results3d_dir}'/'$data_name
i_cam=6
n_frame2draw=$(( $procFrame - $skipFrame - $fps))
videoout_dir=${vid3dout_dir}
python ./visualize_3D.py \
        --config_path ${config_path}\
        --data_name ${data_name} \
        --raw_data_dir ${raw_data_dir}\
        --pickledata_dir ${pickledata_dir}\
        --vidout_dir ${videoout_dir} \
        --i_cam ${i_cam}\
        --n_frame2draw ${n_frame2draw}
