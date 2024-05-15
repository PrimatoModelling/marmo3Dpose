#
# This is same as the "visualize_2_simpleSingleVid.py"
# This is kept for compatibility, so please do not use this function
#
import json
import shutil
import os
import cv2
import numpy as np
from tqdm import tqdm
from math import floor, ceil 
import tempfile
import copy
import matplotlib.pyplot as plt
import math
import platform 
import argparse
import imgstore 

def ellipse_line(img, x1, x2, mrksize, clr):
    if x2[0] - x1[0] == 0:
        ang = 90
    else:
        ang = math.atan((x2[1] - x1[1])/(x2[0] - x1[0])) / math.pi * 180
    cen = ((x1[0] + x2[0])/2, (x1[1] + x2[1])/2)
    d = math.sqrt(math.pow(x2[0] - x1[0], 2) + math.pow(x2[1] - x1[1], 2) )
    
    cv2.ellipse(img, (cen, (d, mrksize), ang), clr, thickness=-1)
    return 0

def clean_kp(kp):
    for i_kp in range(len(kp)):
        # if kp[i_kp][2] < 0.3:
        if kp[i_kp][2] < 0.3:
            kp[i_kp] = None
        else:
            kp[i_kp] = kp[i_kp][0:2]

def add_neckkp(kp):
    if kp[5] is not None and kp[6]is not None:
        d = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
    else:
        d = None
    kp.append(d)
    
def draw_kps(img, kp, mrksize, clr=None):
    
    cm = plt.get_cmap('hsv', 36)

    kp_con = [
            #{'name':'0_2','color':cm(27), 'bodypart':(0,2)},
            #{'name':'0_1','color':cm(31),'bodypart':(0,1)},
            #{'name':'2_4','color':cm(29), 'bodypart':(2,4)},
            #{'name':'1_3','color':cm(33),'bodypart':(1,3)},
            {'name':'0_4','color':cm(29), 'bodypart':(0,4)},
            {'name':'0_3','color':cm(33),'bodypart':(0,3)},
            {'name':'6_8','color':cm(5),'bodypart':(6,8)},
            {'name':'5_7','color':cm(10),'bodypart':(5,7)},
            {'name':'8_10','color':cm(7),'bodypart':(8,10)},
            {'name':'7_9','color':cm(12),'bodypart':(7,9)},
            {'name':'12_14','color':cm(16),'bodypart':(12,14)},
            {'name':'11_13','color':cm(22),'bodypart':(11,13)},
            {'name':'14_16','color':cm(18),'bodypart':(14,16)},
            {'name':'13_15','color':cm(24),'bodypart':(13,15)},
            {'name':'0_18','color':cm(26),'bodypart':(0,18)},
            {'name':'18_6','color':cm(2),'bodypart':(18,6)},
            {'name':'18_5','color':cm(3),'bodypart':(18,5)},
            {'name':'18_17','color':cm(20),'bodypart':(18,17)},
            {'name':'17_12','color':cm(14),'bodypart':(17,12)},
            {'name':'17_11','color':cm(20),'bodypart':(17,11)},
            ]

    kp_clr = [cm(0), cm(32), cm(28), cm(34), cm(30), cm(9), cm(4), cm(11), cm(6), 
              cm(13), cm(8), cm(21), cm(15), cm(23), cm(17), cm(25), cm(19), cm(1)] 

    for i in reversed(range(len(kp))):
        if kp[i] is not None:
            if clr is None:
                c = kp_clr[i]
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, (c[0]*255, c[1]*255, c[2]*255), thickness=-1)
            else:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, (clr[0]*255, clr[1]*255, clr[2]*255), thickness=-1)
            #ax.plot(kp[i][0], kp[i][1], color=kp_clr[i], marker='o', ms=8*mrksize, alpha=1, markeredgewidth=0)
    
    for i in reversed(range(len(kp_con))):
        j1 = kp_con[i]['bodypart'][0]
        j2 = kp_con[i]['bodypart'][1]
        c = kp_con[i]['color']
        if kp[j1]is not None and kp[j2] is not None:
            if clr is None:
                ellipse_line(img, kp[j1], kp[j2], mrksize, (c[0]*255, c[1]*255, c[2]*255))
            else:
                ellipse_line(img, kp[j1], kp[j2], mrksize, (clr[0]*255, clr[1]*255, clr[2]*255))

def procSingleVideo(path_vid,path_json,path_output,n_frame_to_Save,redo=False):
    import os 
    if os.path.exists(path_output) and not redo:
        print(f'Skipping as 2D movie already exists: {path_output}')
        return
    with open(path_json, 'r') as f:
        data = json.load(f)

    n_frame = len(data)   
    
         
    if (n_frame<n_frame_to_Save) | (n_frame_to_Save<1):
        n_frame_to_Save=n_frame

    cnames = ['BLUE', 'RED','INF', 'YELLOW']
    # cnames = ['BLUE', 'Red']
    
    clrs = plt.get_cmap('hsv', 20)

    cap = cv2.VideoCapture(path_vid)
    vw = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'mp4v'), 24, (2048, 1536))
    for i_frame in tqdm(range(n_frame_to_Save)):
        ret, img = cap.read()

        d = data[i_frame]

        for dd in d:
            trkl_id = dd[0]
            bb = dd[1:5]
            kp = copy.deepcopy(dd[5])
            clrinfo = dd[6:8]

            a = min(bb[2]-bb[0],bb[3]-bb[1])
            mrksize = max(int(a/30),1)

            clean_kp(kp)
            add_neckkp(kp)
            draw_kps(img, kp, mrksize, clrs(trkl_id%20))
            clr=clrs(trkl_id%20)
            img =cv2.rectangle(img, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])),(clr[0]*255, clr[1]*255, clr[2]*255),thickness=3)
            
            # frame 
            cv2.putText(img, str(i_frame), (int(10),int(150)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_4)

            # 0.9 
            if clrinfo[1] > 0.9 and int(clrinfo[0]) != 4:
                cv2.putText(img, cnames[int(clrinfo[0])]+":"+str(int(clrinfo[1]*100)), (int(bb[0]), int(bb[1])), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2.0,
                            color=(0, 255, 255),
                            thickness=5,
                            lineType=cv2.LINE_4)
            else : 
                pass 
        vw.write(img)
    vw.release()

if __name__ == '__main__':
    ######
    # path_vid = 'Y:/tkaneko/marmo2hoMotif/dailylife/dailylife_cj425_20210904_090000.23506239/000000.mp4'
    # path_json = './results/dailylife_cj425_20210904_090000/dailylife_cj425_20210904_090000_23506239_0.json'
    # path_output='./output_marmo_cj425_.mp4'
    # path_vid = '/mnt/amakusa4/DataOrg/tkaneko/marmo2hoMotif/dailylife/dailylife_cj425_20220403_110000.23506214/000000.mp4'
    if 0:
        path_vid = 'Y:/tkaneko/marmo2hoMotif/pogz/pogz2/isolation/pairisolation_cj542_cj870_20220704_095812.23506214/000000.mp4'
        path_json = './results2d_2color_tr2/pairisolation_cj542_cj870_20220704_095812/pairisolation_cj542_cj870_20220704_095812_23506214_000000.json'
        path_output='./pairisolation_cj542_cj870_20220704_095812_23506214_000000.mp4'

        path_vid = './demo_results/foodcomp_cj711_cj712_20220720_142016.23506226_000000.mp4'
        path_json = './demo/poseV0p2.json'
        path_output='./demo/poseV0p2.mp4'

        path_vid ='Y:/tkaneko/marmo2hoMotif/pogz/pogz2/id/cj542m_blue_20220720_094645.23506239/000000.mp4'
        path_json ='Z:/kaneko/monkey3d/pipeline/results/2d_v0p31/cj542m_blue_20220720_094645/cj542m_blue_20220720_094645_23506239_000000.json'
        path_output='./demo/poseV0p31.mp4'
        n_frame_to_Save=10000

        camName='23506214'
        path_vid ='Y:/tkaneko/marmo2hoMotif/pogz/pogz2/parenting/family3_pogz2_cj542_cj578_cj870_20220720_104326.'+camName+'/000000.mp4'
        path_json ='Z:/kaneko/monkey3d/pipeline/results/2d_v0p32/family3_pogz2_cj542_cj578_cj870_20220720_104326/family3_pogz2_cj542_cj578_cj870_20220720_104326_'+camName+'_000000.json'
        path_output='./results/video/2d_v0p32/family3_pogz2_cj542_cj578_cj870_20220720_104326_'+camName+'_000000.mp4'
        n_frame_to_Save=10000
    else: 
        parser = argparse.ArgumentParser()     
        parser.add_argument('--path_vid', default='', type=str, help='fullpath for video e.g. ./X.mp4')
        parser.add_argument('--path_json', default='', type=str, help='fullpath for keypoints json')
        parser.add_argument('--path_output', default='', type=str, help='fullpath for output video, e.g. ./test.mp4')
        parser.add_argument('--n_frame_to_save', default=10000, type=int,  help='Number of frames to draw, -1=all')
        parser.add_argument('--redo', default=False, type=bool,  help='redo')
        args = parser.parse_args() 
        path_vid = args.path_vid
        path_json = args.path_json
        path_output= args.path_output
        n_frame_to_Save=int(args.n_frame_to_save)

    procSingleVideo(path_vid,path_json,path_output,n_frame_to_Save, redo=args.redo)