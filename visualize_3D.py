#from lib2to3.pytree import _Results
from distutils.command.install_egg_info import install_egg_info
from genericpath import isdir
import os

import multicam_toolbox as mct

import os
import cv2
import numpy as np
from tqdm import tqdm
from math import floor, ceil 
import yaml
import matplotlib.pyplot as plt
import math
import h5py
import pickle
import imgstore
import scipy.io
import platform
import argparse

def ellipse_line(img, x1, x2, mrksize, clr):
    if x2[0] - x1[0] == 0:
        ang = 90
    else:
        ang = math.atan((x2[1] - x1[1])/(x2[0] - x1[0])) / math.pi * 180
    cen = ((x1[0] + x2[0])/2, (x1[1] + x2[1])/2)
    d = math.sqrt(math.pow(x2[0] - x1[0], 2) + math.pow(x2[1] - x1[1], 2) )
    
    cv2.ellipse(img, (cen, (d, mrksize), ang), clr, thickness=-1)
    return 0

def clean_kp(kp, ignore_score=False, show_as_possible=True):

    cnt = 0
    for i_kp in range(len(kp)):
        if kp[i_kp][2] > 0.3:
            cnt += 1

    for i_kp in range(len(kp)):

        #if i_kp == 1 or i_kp == 2:
        #    kp[i_kp] = None
        #    continue

        if show_as_possible:
            if cnt == 0:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][0]):
                kp[i_kp] = None
            elif kp[i_kp][0] > 3000 or kp[i_kp][0] < -1000 or kp[i_kp][1] > 3000 or kp[i_kp][1] < -1000:
                kp[i_kp] = None
            else:
                kp[i_kp] = kp[i_kp][0:2]
        else:
            if kp[i_kp][2] < 0.3 and not ignore_score:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][2]) and not ignore_score:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][0]):
                kp[i_kp] = None
            elif kp[i_kp][0] > 3000 or kp[i_kp][0] < -1000 or kp[i_kp][1] > 3000 or kp[i_kp][1] < -1000:
                kp[i_kp] = None
            else:
                kp[i_kp] = kp[i_kp][0:2]

def draw_kps(img, kp, mrksize, clr=None):
    
    cm = plt.get_cmap('hsv', 36)

    kp_con = [
            {'name':'0_2','color':cm(27), 'bodypart':(0,2)},
            {'name':'0_1','color':cm(31),'bodypart':(0,1)},
            {'name':'2_4','color':cm(29), 'bodypart':(2,4)},
            {'name':'1_3','color':cm(33),'bodypart':(1,3)},
            #{'name':'0_4','color':cm(29), 'bodypart':(0,4)},
            #{'name':'0_3','color':cm(33),'bodypart':(0,3)},
            {'name':'6_8','color':cm(5),'bodypart':(6,8)},
            {'name':'5_7','color':cm(10),'bodypart':(5,7)},
            {'name':'8_10','color':cm(7),'bodypart':(8,10)},
            {'name':'7_9','color':cm(12),'bodypart':(7,9)},
            {'name':'12_14','color':cm(16),'bodypart':(12,14)},
            {'name':'11_13','color':cm(22),'bodypart':(11,13)},
            {'name':'14_16','color':cm(18),'bodypart':(14,16)},
            {'name':'13_15','color':cm(24),'bodypart':(13,15)},
            {'name':'18_17','color':cm(1),'bodypart':(18,17)},
            {'name':'0_18','color':cm(1),'bodypart':(0,18)},
            {'name':'18_6','color':cm(2),'bodypart':(18,6)},
            {'name':'18_5','color':cm(3),'bodypart':(18,5)},
            {'name':'17_12','color':cm(14),'bodypart':(17,12)},
            {'name':'17_11','color':cm(20),'bodypart':(17,11)}
            ]

    kp_clr = [cm(0), cm(32), cm(28), cm(34), cm(30), cm(9), cm(4), cm(11), cm(6), 
              cm(13), cm(8), cm(21), cm(15), cm(23), cm(17), cm(25), cm(19), cm(1), cm(1)] 

    for i in reversed(range(len(kp))):
        if kp[i] is not None:
            c = kp_clr[i]
            if clr is None:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, (c[0]*255, c[1]*255, c[2]*255), thickness=-1)
            else:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, (clr[0]*255, clr[1]*255, clr[2]*255), thickness=-1)
            #ax.plot(kp[i][0], kp[i][1], color=kp_clr[i], marker='o', ms=8*mrksize, alpha=1, markeredgewidth=0)
    
    for i in reversed(range(len(kp_con))):
        j1 = kp_con[i]['bodypart'][0]
        j2 = kp_con[i]['bodypart'][1]
        c = kp_con[i]['color']
        if kp[j1] is not None and kp[j2] is not None:
            if clr is None:
                ellipse_line(img, kp[j1], kp[j2], mrksize, (c[0]*255, c[1]*255, c[2]*255))
            else:
                ellipse_line(img, kp[j1], kp[j2], mrksize, (clr[0]*255, clr[1]*255, clr[2]*255))

def reproject(config_path, i_cam, p3d):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    id = ID[i_cam]

    path_intrin = os.path.dirname(config_path) + '/cam_intrinsic.h5'
    path_extrin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'

    with h5py.File(path_intrin, mode='r') as f_intrin:
        mtx = f_intrin['/'+str(id)+'/mtx'][()]
        dist = f_intrin['/'+str(id)+'/dist'][()]
        K = f_intrin['/'+str(id)+'/K'][()]
        xi = f_intrin['/'+str(id)+'/xi'][()]
        D = f_intrin['/'+str(id)+'/D'][()]
    with h5py.File(path_extrin, mode='r') as f_extrin:
        rvecs = f_extrin['/'+str(id)+'/rvec'][()]
        tvecs = f_extrin['/'+str(id)+'/tvec'][()]

    pts, _ = cv2.omnidir.projectPoints(np.reshape(p3d, (-1,1,3)), rvecs, tvecs, K, xi[0][0], D)

    return pts[:,0,:]
def procSingleVideo(config_path, data_name, raw_data_dir,result_dir,vidout_dir, 
                    i_cam=7,n_frame2draw=-1,ignore_score=False,show_as_possible=True,redo=False,
                    showID=True,mrksize=5):    
    pickleFile= result_dir + '/kp3d_fxdJointLen.pickle'    
    if (not os.path.exists(pickleFile)):
        pickleFile= result_dir + '/kp3d.pickle'    
    print(pickleFile)
    # -----

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    if os.path.isdir(vidout_dir):            
        videoOutputFile=vidout_dir+'/'+data_name + '_{:d}.mp4'.format(ID[i_cam])
    else:
        videoOutputFile=vidout_dir

    if os.path.exists(videoOutputFile) & (not redo):
        print("Exist:" + videoOutputFile)
        return 
    if not os.path.exists(pickleFile):
        print("Unable to find:" + pickleFile)
        return 
        
    vw = cv2.VideoWriter(videoOutputFile, cv2.VideoWriter_fourcc(*'mp4v'), 24, (800, 600))

    with open(pickleFile, 'rb') as f:
        data = pickle.load(f)

    data_name = os.path.basename(result_dir)
    mdata = raw_data_dir + '/' + data_name + '.' + str(ID[i_cam]) + '/metadata.yaml'
    frame_num = np.load(result_dir + '/' + str(ID[i_cam]) + '/frame_num.npy')
    store = imgstore.new_for_filename(mdata)

    X = data['kp3d']
    S = data['kp3d_score']

    n_animal, n_frame, n_kp, _ = X.shape
    n_cam = len(ID)

    clrs = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]

    if n_frame2draw<0:
        n_frame2draw=n_frame
    if n_frame2draw > n_frame:
        n_frame2draw=n_frame

    for i_frame in tqdm(range(n_frame2draw)):
    #for i_frame in tqdm(range(200)):
        i = frame_num[i_frame]
        img, (frame_number, frame_time) = store.get_image(frame_number=i)

        for i_animal in range(n_animal):
            x = X[i_animal, i_frame, :, :]
            s = S[i_animal, i_frame, :]

            a = (x[5,:] + x[6,:])/2
            x = np.concatenate([x,a[np.newaxis,:]],axis=0)

            a = (s[5] + s[6])/2
            s = np.concatenate([s,a[np.newaxis]],axis=0)

            p = reproject(config_path, i_cam, x)
            I = np.logical_not(x[:,0]==0)
            I = np.logical_and(I, s>0.0)
            p = np.concatenate([p, I[:,np.newaxis]], axis=1)

            kp = p.tolist()
            # mrksize = 5
            clean_kp(kp, ignore_score=ignore_score, show_as_possible=show_as_possible)
            draw_kps(img, kp, mrksize, None)
            #if i_animal < 4:
            #    draw_kps(img, kp, mrksize, clrs[i_animal])
            #else: 
            #    draw_kps(img, kp, mrksize, (0,0,0))
            idDrawX=p[:,0].min()
            idDrawY=p[:,1].min()
                        
            if showID==True:
                if (not math.isnan(idDrawX)) & (not math.isnan(idDrawY)):                
                    cv2.putText(img, str(i_animal), (int(idDrawX),int(idDrawY)), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2.0,
                                    color=(0, 255, 255),
                                    thickness=5,
                                    lineType=cv2.LINE_4)
        cv2.putText(img, str(i_frame), (int(10),int(150)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_4)
        img = cv2.resize(img, (800,600))

        vw.write(img)

    vw.release()

if __name__ == '__main__':
    if 0:
        config_path = './calib/marmo/config.yaml'
        data_name = 'dailylife_cj425_20220313_130000'
        if platform.system()=='Windows':
            raw_data_dir ='Y:/tkaneko/marmo2hoMotif/dailylife'
        else:
            raw_data_dir= '/mnt/amakusa4/DataOrg/tkaneko/marmo2hoMotif/dailylife'
        i_cam = 7 # camera index for visualization    
        pickledata_dir = './results/3d_v0p3/' + data_name    
        vidout_dir='./results/video'    
        n_frame2draw=-1
        # n_frame2draw=1800
    else: 
        parser = argparse.ArgumentParser()   
        parser.add_argument('--config_path', default='./calib/marmo/config.yaml', help='Fullpath of config file')
        parser.add_argument('--data_name', default='', type=str, help='session name, e.g., dailylife_cj425_20220402_110000')
        parser.add_argument('--raw_data_dir',     default='cuda:0',      help='Root directory of motif videos')
        parser.add_argument('--pickledata_dir', default='./results/3d', help='Root directory to save 3D results')
        parser.add_argument('--vidout_dir',      default='./results/video',   help='Root directory to save video, or fullpath of video')

        parser.add_argument('--i_cam', default=7, type=int, help='Camera index, e.g., 7')
        parser.add_argument('--n_frame2draw', default=-1, type=int,  help='Number of frames to draw, -1=all')
        
        parser.add_argument('--redo', default=False, type=bool,  help='redo 0 or 1')
        parser.add_argument('--showID', default=True, type=int,  help='show id or not 0 or 1')
        parser.add_argument('--mrksize', default=5, type=int, help='marker size; default=5')

        args = parser.parse_args() 
        config_path = args.config_path
        data_name        = args.data_name
        raw_data_dir = args.raw_data_dir
        pickledata_dir = args.pickledata_dir
        vidout_dir=args.vidout_dir
        redo=args.redo

        i_cam = args.i_cam
        n_frame2draw = args.n_frame2draw
        if args.showID==0:
            showID=False
        else:
            showID=True        
        mrksize = args.mrksize
    ignore_score = False
    show_as_possible = True
    print('showID',args.showID)
    procSingleVideo(config_path, data_name, raw_data_dir,pickledata_dir,vidout_dir,
                    i_cam=i_cam,n_frame2draw=n_frame2draw,
                    ignore_score=ignore_score,show_as_possible=show_as_possible,redo=redo,showID=showID,mrksize=mrksize)