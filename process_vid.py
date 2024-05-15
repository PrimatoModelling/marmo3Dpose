import os
import struct
import warnings
from argparse import ArgumentParser

import pickle
from scipy.fft import skip_backend
from tqdm import tqdm
import cv2
import time
import json
import numpy as np
import torch 

from mmcv import imcrop

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

from mmtrack.apis import inference_mot
from mmtrack.apis import init_model as init_tracking_model

from mmcls.apis import inference_model as inference_cls_model
from mmcls.apis import init_model as init_cls_model

class dummyStruct(object):
    pass

def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    :param mmtracking_results:
    :return: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if 'track_bboxes' in mmtracking_results:
        tracking_results = mmtracking_results['track_bboxes'][0]
    elif 'track_results' in mmtracking_results:
        tracking_results = mmtracking_results['track_results'][0]

    for track in tracking_results:
        person = {}
        person['track_id'] = int(track[0])
        person['bbox'] = track[1:]
        person_results.append(person)
    return person_results

# def main(cfg,redo=False):
def main(cfg,redo=False,doDet=1,doPose=1,doID=1):
    vid_path = cfg.vid_path
    out_json_path = cfg.out_json_path
    out_vid_path = cfg.out_vid_path
    tracking_config = cfg.tracking_config # 'model/track/jm_bytetrack_yolox.py'
    pose_config = cfg.pose_config #  'model/pose/tk_hrnet_w32_256x256_nopretrained.py'
    pose_checkpoint = cfg.pose_checkpoint # 'weight/pose.pth'
    id_config = cfg.id_config # 'model/id/tk_resnet50_8xb32_in1k.py'
    id_checkpoint = cfg.id_checkpoint # 'weight/id.pth'
    # 
    if os.path.exists(out_json_path) & (not redo):
        print('Exist proc vid:'+out_json_path)
        return 

    bbox_thr = 0.3
    vis_radius = 4
    vis_thickness = 2
    vis_kpt_thr = 0.3

    tracking_model = init_tracking_model(
        tracking_config, None, device=cfg.device.lower())

    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=cfg.device.lower())

    id_model = init_cls_model(id_config, id_checkpoint, device=cfg.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(vid_path)
    assert cap.isOpened(), f'Faild to load video file {vid_path}'

    if cfg.procFrame<1:
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else: 
        n_frame=cfg.procFrame
    skip_frame=cfg.skipFrame
    
    if out_vid_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(out_vid_path, fourcc,
            fps, size)

    return_heatmap = False
    output_layer_names = None

    torch.set_num_threads(6)

    result = []
    isSkipping=1
    for frame_id in tqdm(range(n_frame)):
        if skip_frame>frame_id:       ### フレームをとばす場合
            pose_results = []   ##ダミーデータを挿入
        else:
            if isSkipping:
                isSkipping=0
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
            # frame load
            s = time.time()
            flag, img = cap.read()
            if not flag:
                break
                # print('frame load (ms):', (time.time()-s)*1000)

            # detection & tracking
            s = time.time()
            mmtracking_results = inference_mot(
                tracking_model, img, frame_id=frame_id)
            person_results = process_mmtracking_results(mmtracking_results)
                # print('detection+tracking (ms):', (time.time()-s)*1000)

            # keypoint estimation
            s = time.time()
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
                # print('pose (ms):', (time.time()-s)*1000)

            # check id
            s = time.time()
                #print(len(pose_results))
                #if len(pose_results)==0:
                #    print('no detection!')
            if len(pose_results) > 0:
                bboxes = []
                for p in pose_results:
                    bboxes.append(p['bbox'][:4])
                bboxes = np.array(bboxes)
                patches = imcrop(img, bboxes)
                for i_pose, patch in enumerate(patches):
                    id_result = inference_cls_model(id_model, patch)
                    pose_results[i_pose]['id'] = id_result
                # print('id (ms):', (time.time()-s)*1000)
        result.append(pose_results)

        if out_vid_path is not None:
            vis_img = vis_pose_tracking_result(
                pose_model,
                img,
                pose_results,
                radius=vis_radius,
                thickness=vis_thickness,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=vis_kpt_thr,
                show=False)
            if len(pose_results) > 0:               
                for i_box, tt in enumerate(pose_results):                                                                     
                    if tt['id']['pred_score'] > 0.97 and int(tt['id']['pred_label']) != 4:
                        cnames = ['B', 'I','R', 'X']
                        cv2.putText(vis_img, cnames[int(tt['id']['pred_label'])], (int(tt['bbox'][0]), int(tt['bbox'][1])), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2.0,
                                    color=(0, 255, 255),
                                    thickness=5,
                                    lineType=cv2.LINE_4)

            videoWriter.write(vis_img)

    cap.release()
    if out_vid_path is not None:
        videoWriter.release()

    #with open('result.pickle','wb') as f:
    #    pickle.dump(result,f)

    data = []
    for i_frame, t in enumerate(result):
        data.append([])
        for i_box, tt in enumerate(t):
            d = [tt['track_id'], tt['bbox'][0], tt['bbox'][1], tt['bbox'][2], tt['bbox'][3], 
                tt['keypoints'].tolist(), int(tt['id']['pred_label']), tt['id']['pred_score']]
            data[i_frame].append(d)
            #T2[i_frame][i_box] = [tt[0], tt[1], tt[2], tt[3], tt[4], K[i_frame][i_box], C[i_frame, i_box, 0], C[i_frame, i_box, 1]]

    with open(out_json_path, 'w') as f:
        json.dump(data, f)

def proc(vid_path,out_json_path,device_str,redo=False,doDet=1,doPose=1,doID=1,procFrame=-1,skipFrame=-1,
         tracking_config = 'model/track/jm_bytetrack_yolox.py',
         pose_config     = 'model/pddose/tk_hrnet_w32_256x256_nopretrained.py',
         pose_checkpoint = 'weight/pose.pth',
         id_config       = 'model/id/tk_resnet50_8xb32_in1k.py',
         id_checkpoint   = 'weight/id.pth'):    
    cfg =dummyStruct()    
    cfg.vid_path =vid_path
    cfg.out_json_path = out_json_path
    cfg.device = device_str
    cfg.out_vid_path = None
    cfg.tracking_config = tracking_config 
    cfg.pose_config = pose_config
    cfg.pose_checkpoint = pose_checkpoint
    cfg.id_config = id_config
    cfg.id_checkpoint = id_checkpoint
    cfg.procFrame=procFrame
    cfg.skipFrame=skipFrame    
    main(cfg,redo,doDet,doPose,doID)

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--vid_path', default='./demo/pairisolation_cj542_cj835_20211222_161602_23506237_000000.mp4', type=str, help='Video path')
    parser.add_argument('--vid_path', default='./demo/family3_pogz2_cj542_cj578_cj870_20220725_100820_23506214_000000.mp4', type=str, help='Video path')    
    parser.add_argument('--out_json_path', default='./tmptest_RBIv0p9full_.json', type=str, help='Result json path')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--redo', default=False, type=bool, help='redo and overwrite')
    parser.add_argument('--procFrame', default=1000, type=int, help='-1 is all')
    parser.add_argument('--skipFrame', default=-1, type=int, help='-1 is all')

    parser.add_argument('--out_vid_path',    default="tmptest_RBIv0p8full.mp4", type=str, help='Video output fullpath if you need video output, leave empty not to save')
    parser.add_argument('--tracking_config', default='model/track/tk_bytetrack_yolox_v0p8.py', type=str, help='e.g., model/track/jm_bytetrack_yolox.py')
    parser.add_argument('--pose_config',     default='model/pose/tk_hrnet_w32_256x256_V0p4.py', type=str, help='e.g., model/pose/tk_hrnet_w32_256x256_nopretrained.py')
    parser.add_argument('--pose_checkpoint', default='weight/pose_v0p4.pth', type=str, help='e.g., weight/pose.pth')
    parser.add_argument('--id_config',       default='model/id/tk_resnet50_8xb32_RBI_v4.py', type=str, help='e.g., model/id/tk_resnet50_8xb32_in1k.py')
    parser.add_argument('--id_checkpoint',    default='weight/id_marmo_RBI_v4.pth', type=str, help='e.g., weight/id.pth')    
    args = parser.parse_args()

    cfg =dummyStruct()    
    cfg.vid_path = args.vid_path
    cfg.out_json_path = args.out_json_path
    cfg.device= args.device
    cfg.out_vid_path = args.out_vid_path
    cfg.tracking_config = args.tracking_config
    cfg.pose_config = args.pose_config
    cfg.pose_checkpoint = args.pose_checkpoint
    cfg.id_config = args.id_config
    cfg.id_checkpoint = args.id_checkpoint
    cfg.procFrame = args.procFrame
    cfg.skipFrame=args.skipFrame    
    main(cfg)