_base_ = './tk_yolox_s_8x8_300e_coco_V0p11.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

#プレトレインのモデルは以下に入力
load_from = './checkpoints/tk_yolox_l_8x8_300e_coco_V0p10.pth'