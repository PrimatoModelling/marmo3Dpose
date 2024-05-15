import process_vid 
import argparse
import os 
import platform 
import glob
import  process_vid
import argparse
import yaml
import math
def proc(data_name,config_path,raw_data_dir,label2d_dir,device_str,
         procFrame=-1,
         skipFrame=-1,
         tracking_config = 'model/track/jm_bytetrack_yolox.py',
         pose_config     = 'model/pose/tk_hrnet_w32_256x256_nopretrained.py',
         pose_checkpoint = 'weight/pose.pth',
         id_config       = 'model/id/tk_resnet50_8xb32_in1k.py',
         id_checkpoint   = 'weight/id.pth'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    camIDs = cfg['camera_id']

    session = data_name
    saveDir=os.path.join(label2d_dir,session)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    for camID in camIDs:
        vidFolder=os.path.join(raw_data_dir,session+'.'+str(camID))        
        # print(vidFolder)
        vidList=glob.glob(vidFolder+'/*.mp4')
        if procFrame>0:
            # nvid=(procFrame//10000)+1            
            nvid=math.ceil(procFrame/10000)
            vidList=vidList[0:nvid]
        for ivid,tmp in enumerate(vidList):       
            currSkipFrame=skipFrame-(10000*ivid)            
            currProcFrame=procFrame-(10000*ivid)      
            if currProcFrame>10000:
                currProcFrame=10000     
            currVid=vidList[ivid]        
            movID=os.path.split(currVid)[-1].split('.')[0]
            vid_path =currVid
            out_json_path=os.path.join(label2d_dir,session,session+'_'+str(camID)+'_'+movID+'.json')            
            process_vid.proc(vid_path,out_json_path,device_str,
            procFrame=currProcFrame,
            skipFrame=currSkipFrame,
            tracking_config=tracking_config,
            pose_config=pose_config, pose_checkpoint=pose_checkpoint,
            id_config=id_config,id_checkpoint=id_checkpoint)    
            
if __name__ == '__main__':
    if 0:
        if platform.system()=='Windows':
            raw_data_dir ='Y:/tkaneko/marmo2hoMotif/dailylife'
        else:
            raw_data_dir= '/mnt/amakusa4/DataOrg/tkaneko/marmo2hoMotif/dailylife'
            label2d_dir='./results'
            device_str='cuda:1'
            config_path='./calib/marmo/config.yaml'    
            data_name='dailylife_cj425_20211009_110000'
    else :
        parser = argparse.ArgumentParser()                
        parser.add_argument('--config_path',  default='./calib/marmo/config.yaml', help='Fullpath of config file')
        parser.add_argument('--data_name',    default='', type=str,       help='session name, e.g., dailylife_cj425_20220402_110000')
        parser.add_argument('--raw_data_dir', default='cuda:0',           help='Root directory of motif videos')
        parser.add_argument('--label2d_dir',  default='./results',        help='Root directory of 2D keypoints')        
        parser.add_argument('--device_str',   default='cuda:1', type=str, help='GPU device, e.g."cuda:0,1"')
        parser.add_argument('--procFrame',    default='-1',    type=str, help='nframe to analyze or -1 for all')
        parser.add_argument('--skipFrame',    default='-1',    type=str, help='nframe to skip, or 0or-1 for 0 skip')

        parser.add_argument('--tracking_config', default='model/track/tk_bytetrack_yolox_v0p3.py',  type=str, help='e.g."model/track/tk_bytetrack_yolox_v0p3.py"')
        parser.add_argument('--pose_config',     default='model/pose/tk_hrnet_w32_256x256_V0p3.py', type=str, help='e.g."model/pose/tk_hrnet_w32_256x256_V0p3.py"')
        parser.add_argument('--pose_checkpoint', default='weight/pose_v0p3.pth',                    type=str, help='e.g."weight/pose_v0p3.pth"')
        parser.add_argument('--id_config',       default='model/id/tk_resnet50_8xb32_in1k.py',      type=str, help='e.g."model/id/tk_resnet50_8xb32_in1k.py"')
        parser.add_argument('--id_checkpoint',   default='weight/id.pth',                           type=str, help='e.g."weight/id.pth"')
        args = parser.parse_args()

        data_name    = args.data_name
        config_path  = args.config_path
        raw_data_dir = args.raw_data_dir
        label2d_dir  = args.label2d_dir
        device_str   = args.device_str   
        procFrame    = int(args.procFrame)
        skipFrame    = int(args.skipFrame)

        tracking_config = args.tracking_config
        pose_config =args.pose_config
        pose_checkpoint=args.pose_checkpoint
        id_config=args.id_config
        id_checkpoint=args.id_checkpoint
    
    proc(data_name,config_path,raw_data_dir,label2d_dir,device_str,
        procFrame,
        skipFrame,
        tracking_config=tracking_config,
        pose_config=pose_config, pose_checkpoint=pose_checkpoint,
        id_config=id_config,id_checkpoint=id_checkpoint)