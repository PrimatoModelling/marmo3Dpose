from numpy import False_
from pandas import date_range
import reconst_single_step1 as step1
import reconst_multi_step2 as step2
import reconst_multi_step3 as step3
import reconst_single_step3 as step4
import argparse
import platform
import os
def proc(data_name,results3d_dir, config_path,
         raw_data_dir, label2d_dir,fps,t_intv,
         n_kp,thr_kp_detection,config_3d_toml='./config_tmpl.toml',calib_3d_toml='./calibration_tmpl.toml'):
    print("Proc:%s" % (data_name))
    step1.proc(data_name,results3d_dir, raw_data_dir, label2d_dir,fps,t_intv)    
    # step2.proc(data_name,results3d_dir, config_path, n_kp,thr_kp_detection)
    step2.proc(data_name,results3d_dir,raw_data_dir,config_path)  
    save_vid = False
    save_vid_cam = 1
    step3.proc(data_name,results3d_dir,raw_data_dir,config_path,save_vid,save_vid_cam)     
    redo=False
    step4.proc(data_name,results3d_dir, config_path, n_kp,
        redo=redo,config_3d_toml=config_3d_toml,calib_3d_toml=calib_3d_toml)        
    
if __name__ == '__main__':
    if 0: 
        # config 
        config_path = './calib/marmo/config.yaml'    

        # dirs 
        label2d_dir = './results2d_2color_tr2'
        results3d_dir='./results3D_2color_tr2'
        if platform.system()=='Windows':
            raw_data_dir ='Y:/tkaneko/marmo2hoMotif/foodcomp_collar'
        else:
            raw_data_dir= '/mnt/amakusa4/DataOrg/tkaneko/marmo2hoMotif/foodcomp_collar'

        # params 
        fps = 24
        t_intv = [10, 300]# use None to analyze all
        n_kp = 18
        thr_kp_detection = 0.5        

        # session
        data_name = 'foodcomp_cj711_cj712_20220606_142445'
    else:
        parser = argparse.ArgumentParser()    

        # config 
        parser.add_argument('--config_path', default='./calib/marmo/config.yaml', help='Fullpath of config file')
        parser.add_argument('--config_3d_toml', default='./config_tmpl.toml'    , help='Fullpath of config_3d_toml')
        parser.add_argument('--calib_3d_toml', default='./calibration_tmpl.toml', help='Fullpath of calib_3d_toml')

        # params  
        parser.add_argument('--fps',              default=24, type=int,       help='fsp')
        parser.add_argument('--t_intv',           default='10,300', type=str, help='time interval to analyze, e.g. "10, 300" or "None"')
        parser.add_argument('--n_kp',             default=18, type=int,       help='number of keypoints')
        parser.add_argument('--thr_kp_detection', default=0.5, type=float,    help='threshold of key-point detection')     

        # dirs 
        parser.add_argument('--results3d_dir', default='./results3D', help='Root directory to save 3D results')
        parser.add_argument('--raw_data_dir',     default='cuda:0',      help='Root directory of motif videos')
        parser.add_argument('--label2d_dir',      default='./results',   help='Root directory of 2D keypoints')

        # session
        parser.add_argument('--data_name', default='', type=str, help='session name, e.g., dailylife_cj425_20220402_110000')

        args = parser.parse_args()

        data_name        = args.data_name
        raw_data_dir     = args.raw_data_dir
        label2d_dir      = args.label2d_dir
        results3d_dir = args.results3d_dir
        fps         = args.fps
        config_path = args.config_path
        config_3d_toml=args.config_3d_toml
        calib_3d_toml=args.calib_3d_toml
        n_kp        = args.n_kp
        thr_kp_detection=args.thr_kp_detection
        if args.t_intv.lower()=='none':
            t_intv=None
        else:
            t_intv    = [0,0]
            t_intv[0] = int(args.t_intv.split(',')[0])
            t_intv[1] = int(args.t_intv.split(',')[1])
    proc(data_name,results3d_dir, config_path, raw_data_dir, label2d_dir,fps,t_intv, n_kp,thr_kp_detection,
        config_3d_toml=config_3d_toml,calib_3d_toml=calib_3d_toml)
    # step1.proc(data_name, results3d_dir, raw_data_dir, label2d_dir,fps,t_intv)
    # step2.proc(data_name, results3d_dir, config_path, n_kp,thr_kp_detection)
    # step3.proc(data_name, results3d_dir, config_path, n_kp)
