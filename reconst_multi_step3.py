# step b3
# connect matches to get trace with ID

# 20220804 - 
# RMSE thresholds (500, 400)をマーモ用に変更
# 要変更箇所は、"2022/8/4"で検索
# 
# conf_thr = 0.9 -> 0.95
#

import os
import os.path as osp
import pickle
import sys
import time
import multicam_toolbox as mct
import os.path  as osp
import cv2
import numpy as np
import matplotlib
import copy
import h5py
import yaml
import math
import matplotlib.pyplot as plt
import itertools
import json
import imgstore
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import argparse

def get_camparam(config_path):
    camparam = {}

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    camparam['camera_id'] = ID

    path_extrin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'

    K = []
    xi = []
    D = []
    rvecs = []
    tvecs = []
    for i_cam, id in enumerate(ID):
        with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_intrin:
            K.append(f_intrin['/'+str(id)+'/K'][()])
            xi.append(f_intrin['/'+str(id)+'/xi'][()])
            D.append(f_intrin['/'+str(id)+'/D'][()])
        
        with h5py.File(path_extrin, mode='r') as f_extrin:
            rvecs.append(f_extrin['/'+str(id)+'/rvec'][()])
            tvecs.append(f_extrin['/'+str(id)+'/tvec'][()])

    camparam['K'] = K
    camparam['xi'] = xi
    camparam['D'] = D
    camparam['rvecs'] = rvecs
    camparam['tvecs'] = tvecs


    pmat = []
    for i_cam, id in enumerate(ID):
        with h5py.File(path_extrin, mode='r') as f_extrin:
            rvecs = f_extrin['/'+str(id)+'/rvec'][()]
            tvecs = f_extrin['/'+str(id)+'/tvec'][()]
            rmtx, jcb = cv2.Rodrigues(rvecs)
            R = np.hstack([rmtx, tvecs])
            pmat.append(R)

    camparam['pmat'] = pmat

    return camparam

def calc_3dpose(kp_2d, config_path,camparam=None):
    n_cam, n_kp, _ = kp_2d.shape

    pos_2d = []
    for i_cam in range(n_cam):
        pos_2d.append(kp_2d[i_cam,:,:2])
    pos_2d_undist = mct.undistortPoints(config_path, pos_2d, omnidir=True, camparam=camparam)

    frame_use = np.ones((n_kp, n_cam), dtype=bool)
    for i_kp in range(n_kp):
        for i_cam in range(n_cam):
            if np.isnan(pos_2d[i_cam][i_kp, 0]):
                frame_use[i_kp, i_cam] = False
            if kp_2d[i_cam,i_kp,2] < 0.3:
                frame_use[i_kp, i_cam] = False

    kp_3d = mct.triangulatePoints(config_path, pos_2d_undist, frame_use, True, camparam=camparam)

    return kp_3d

def calc_dist_pose(p1, p2):
    
    d = np.sum((p1-p2)**2, axis=1)
    d = d[np.logical_not(np.isnan(d))]
    if d.shape[0] == 0:
        rmse = 1.0e8
    rmse = np.sqrt(np.sum(d)/d.shape[0])

    if np.isnan(rmse):
        rmse = 1.0e8

    return rmse

def connect_keyframe(config_path, result_dir):

    camparam = get_camparam(config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    T = []
    for i_cam, id in enumerate(ID):
        with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
            data = json.load(f)
        T.append(data)

    with open(result_dir + '/match_keyframe.pickle', 'rb') as f:
        result_keyframe = pickle.load(f)


    n_keyframe = len(result_keyframe)

    n_cam = 8
    n_kp = 18

    C = []

    for i_kf in tqdm(range(1, n_keyframe)):
        i_frame = result_keyframe[i_kf]['frame']
        bbox_pre = result_keyframe[i_kf-1]['bcomb']
        bbox_crnt = result_keyframe[i_kf]['bcomb']

        n_person_pre = len(bbox_pre)
        n_person_crnt = len(bbox_crnt)

        #print(n_person_pre, n_person_crnt)

        P2d_pre = np.zeros([n_person_pre, n_cam, n_kp, 3])
        P2d_pre[:,:,:,:] = np.nan
        P2d_crnt  = np.zeros([n_person_crnt, n_cam, n_kp, 3])
        P2d_crnt[:,:,:,:] = np.nan


        for i_cam in range(n_cam):
            TT = T[i_cam][i_frame]
            for tt in TT:
                bbox_id = tt[0]
                kp = np.array(tt[5])

                for i_person in range(n_person_pre):
                    if bbox_id == bbox_pre[i_person][i_cam]:
                        P2d_pre[i_person, i_cam, :,:] = kp
                
                for i_person in range(n_person_crnt):
                    if bbox_id == bbox_crnt[i_person][i_cam]:
                        P2d_crnt[i_person, i_cam, :,:] = kp
        
        P3d_pre = np.zeros([n_person_pre, n_kp, 3])
        for i_person in range(n_person_pre):
            P3d_pre[i_person, :, :] = calc_3dpose(P2d_pre[i_person, :,:,:], config_path,camparam=camparam)

        P3d_crnt = np.zeros([n_person_crnt, n_kp, 3])
        for i_person in range(n_person_crnt):
            P3d_crnt[i_person, :, :] = calc_3dpose(P2d_crnt[i_person, :,:,:],config_path,camparam=camparam)

        rmse = np.zeros([n_person_pre, n_person_crnt], dtype=float)
        for i_pre in range(n_person_pre):
            for i_crnt in range(n_person_crnt):
                rmse[i_pre, i_crnt] = calc_dist_pose(P3d_pre[i_pre, :, :], P3d_crnt[i_crnt, :, :] )

        #print(rmse)

        row_ind, col_ind = linear_sum_assignment(rmse)

        c = []
        for i in range(len(row_ind)):
            if rmse[row_ind[i], col_ind[i]] < 150: # 2022/8/4 要変更 キーフレーム間でtrackletをつなげる際、所定(500mm -> 150)のRMSE未満でなければ除外; 
                c.append([row_ind[i], col_ind[i]])

        C.append(c)

    with open(result_dir + '/keyframe_connection.pickle', 'wb') as f:
        pickle.dump(C, f)


    """for i_person in range(n_person_pre):
        X = P3d_pre[i_person, :, :] 
        plt.plot(X[:,0], X[:,2], 'r.-')

    for i_person in range(n_person_crnt):
        X = P3d_crnt[i_person, :, :] 
        plt.plot(X[:,0], X[:,2], 'b.-')

    plt.show()"""

def reproject(i_cam, p3d, camparam=None):

    if camparam is None:

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
    else:
        K = camparam['K'][i_cam]
        xi = camparam['xi'][i_cam]
        D = camparam['D'][i_cam]
        rvecs = camparam['rvecs'][i_cam]
        tvecs = camparam['tvecs'][i_cam]

    pts, _ = cv2.omnidir.projectPoints(np.reshape(p3d, (-1,1,3)), rvecs, tvecs, K, xi[0][0], D)

    return pts[:,0,:]

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

def to_intv(I):
    I = np.array(I, dtype=int)

    if I[-1] == 1:
        I = np.append(I, 0)
        
    d = np.diff(np.append(np.array([0]), I))

    start = np.where(d == 1)
    stop = np.where(d == -1)

    start = start[0]
    stop = stop[0]
    intv = np.array([start, stop]).T

    return intv

def div_3dtracklet(Trk, Cid):

    n_cam = 8

    unassigned = []
    assigned = []
    Intv = {}
    for k in Trk.keys():

        if np.sum(Cid[k]>=0) == 0:
            unassigned.append(k)
        else:
            assigned.append(k)
            
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    last_key = max(list(Trk.keys()))

    for k in assigned:

        intv = Intv[k]
        cid = np.unique(Cid[k][intv[0]:intv[1]])

        if cid.shape[0] > 1:

            #print(k, intv)
            n_frame = Cid[k].shape[0]

            for cid2 in cid:
                
                A = np.zeros(n_frame, dtype=bool)
                A[intv[0]:intv[1]] = True
                I = to_intv(np.logical_and(Cid[k]==cid2, A))
                #print('intvs:', I, cid2)

                for i in I:
                    C = -np.ones(n_frame, dtype=int)
                    C[i[0]:i[1]+1] = cid2
                    trk = -np.ones([n_frame, n_cam], dtype=int)
                    trk[i[0]:i[1]+1,:] = Trk[k][i[0]:i[1]+1,:]
                    last_key += 1
                    Cid[last_key] = C
                    Trk[last_key] = trk

            Trk.pop(k)
            Cid.pop(k)

    return Trk, Cid

def dilate_track(Trk, Cid, Cid_conflevel, P3d, conflevel, rmse_thr=200.0): 
    
    ### check existing assignment
    unassigned = []
    assigned = []
    Intv = {}
    for k in Trk.keys():
        if np.sum(Cid[k]>=0) == 0:
            unassigned.append(k)
        else:
            assigned.append(k)
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    ### sort from longer to shorter
    intv_len = []
    for k in unassigned:
        intv_len.append(Intv[k][1]-Intv[k][0])
    intv_len = np.array(intv_len)
    I = np.argsort(intv_len)
    I = I[-1::-1]
    unassigned = np.array(unassigned, dtype=int)
    unassigned = unassigned[I].tolist()

    n_frame = Trk[assigned[0]].shape[0]

    ### dilate_track
    for k in unassigned:
        intv = Intv[k]
        t_s = Trk[k][intv[0],:]
        t_s[t_s==-1] = -2
        t_s = t_s[np.newaxis,:]
        t_e = Trk[k][intv[1],:]
        t_e[t_e==-1] = -2
        t_e = t_e[np.newaxis,:]
        #print(k, intv)

        for k2 in assigned:
            #print(Trk[k2][max(intv[0]-120, 0):intv[0]].shape, t_s.shape)
            
            # check if the overlap
            #cog_u = np.nanmean(P3d[k][intv[0]:intv[1],:,:], axis=1)
            #cog_a = np.nanmean(P3d[k2][intv[0]:intv[1],:,:], axis=1)
            cog_u = P3d[k][intv[0]:intv[1],:]
            cog_a = P3d[k2][intv[0]:intv[1],:]
            d = np.sum((cog_u - cog_a)**2, axis=1)
            I = np.logical_not(np.isnan(d))
            if np.sum(I) > 12:
                d = d[I]
                rmse = np.sqrt(np.sum(d)/d.shape[0])
                if rmse > rmse_thr:
                    continue
            
            # check edge1
            chk_s = np.sum(Trk[k2][max(intv[0]-120, 0):intv[0]] == t_s, axis=0)
            cid = np.unique(Cid[k2][max(intv[0]-120, 0):intv[0]])
            
            if np.sum(chk_s) >= 5 and cid.shape[0] == 1:
                cid = cid[0]
                flag_alreadyexist = False
                for k3 in assigned:
                    if k2 == k3:
                        continue
                    a = np.zeros(n_frame, dtype=int)
                    a[intv[0]:intv[1]] = 1
                    a[Intv[k3][0]:Intv[k3][1]] += 1
                    if np.sum(a==2) > 0:
                        if np.sum(Cid[k3][intv[0]:intv[1]]==cid) > 0:
                            flag_alreadyexist = True
                            #print('exist:', k,k2,k3)

                #print(cid, flag_alreadyexist)

                if flag_alreadyexist == False:
                    #print('changed', k, k2, cid, chk_s)
                    Cid[k][:] = cid
                    assigned.append(k)
                    Cid_conflevel[k] = conflevel
                    break

            # check edge2
            chk_e = np.sum(Trk[k2][intv[1]:min(intv[1]+120,n_frame)] == t_e, axis=0)
            cid = np.unique(Cid[k2][intv[1]:min(intv[1]+120,n_frame)])
            
            if np.sum(chk_e) >= 5 and cid.shape[0] == 1:
                cid = cid[0]
                flag_alreadyexist = False
                for k3 in assigned:
                    if k2 == k3:
                        continue
                    a = np.zeros(n_frame, dtype=int)
                    a[intv[0]:intv[1]] = 1
                    a[Intv[k3][0]:Intv[k3][1]] += 1
                    if np.sum(a==2) > 0:
                        if np.sum(Cid[k3][intv[0]:intv[1]]==cid) > 0:
                            flag_alreadyexist = True
                            #print('exist:', k,k2,k3)

                #print(cid, flag_alreadyexist)

                if flag_alreadyexist == False:
                    #print('changed', k, k2, cid, chk_e)
                    Cid[k][:] = cid
                    assigned.append(k)
                    Cid_conflevel[k] = conflevel
                    break

    return Trk, Cid, Cid_conflevel

def assign_lastone(Trk, Cid, Cid_conflevel, P3d):
    conflevel = 100
    n_animal = 4
    thr = 0.4

    ### check existing assignment
    unassigned = []
    assigned = []
    Intv = {}
    for k in Trk.keys():
        if np.sum(Cid[k]>=0) == 0:
            unassigned.append(k)
        else:
            assigned.append(k)
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    ### sort from longer to shorter
    intv_len = []
    for k in unassigned:
        intv_len.append(Intv[k][1]-Intv[k][0])
    intv_len = np.array(intv_len)
    I = np.argsort(intv_len)
    I = I[-1::-1]
    unassigned = np.array(unassigned, dtype=int)
    unassigned = unassigned[I].tolist()

    n_frame = Trk[assigned[0]].shape[0]
    A = np.zeros([n_frame, n_animal])
    for k in assigned:
        for i_c in range(n_animal):
            A[Intv[k][0]:Intv[k][1],i_c] += Cid[k][Intv[k][0]:Intv[k][1]] == i_c
    A = A>0

    #for i_c in range(4):
    #    plt.plot(A[:,i_c]+i_c*5)
    #plt.show()

    for k in unassigned:
        intv = Intv[k]
        a = A[intv[0]:intv[1],:]
        a = np.sum(a, axis=0)/a.shape[0]
        a = a > thr
        #print('1:', k, intv)
        if np.sum(a) != n_animal-1:
            continue

        #print('2:',k, intv)
        cog_u = P3d[k][intv[0]:intv[1],:]

        flag_overlap = False
        for k2 in assigned:

            cog_a = P3d[k2][intv[0]:intv[1],:]
            d = np.sum((cog_u - cog_a)**2, axis=1)
            I = np.logical_not(np.isnan(d))
            if np.sum(I) > 12:
                d = d[I]
                rmse = np.sqrt(np.sum(d)/d.shape[0])
                if rmse < 120:  # 2022/8/4 要変更   あるtrackletが既にassignされているtrackletと重なっているかどうかの判定(400->120)
                    flag_overlap = True
                    #print(k2, np.unique(Cid[k2]), Intv[k2])

        if flag_overlap:
            continue

        #print('3:',k, intv)
        
        cid = np.argwhere(np.logical_not(a))[0]
        Cid[k][:] = cid
        Cid_conflevel[k] = conflevel
        conflevel += 1

    return Trk, Cid, Cid_conflevel

def analyze_cid(config_path, result_dir, rmse_thr=200.0):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    n_cam = len(ID)

    T = []
    for i_cam, id in enumerate(ID):
        with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
            data = json.load(f)
        T.append(data)

    with open(result_dir + '/match_keyframe.pickle', 'rb') as f:
        result_keyframe = pickle.load(f)

    with open(result_dir + '/keyframe_connection.pickle', 'rb') as f:
        result_keyframe_connection = pickle.load(f)

    ### merge connected traces
    n_kf = len(result_keyframe)
    n_frame = result_keyframe[-1]['frame']
 
    crnt_ids = np.arange(len(result_keyframe[0]['bcomb']), dtype=int)
    if len(crnt_ids)<1: # By tkaneko at 20221003
        cnt=0 # By tkaneko at 20221003
    else:
        cnt = max(crnt_ids)+1

    Trk = {}
    for i_kf in tqdm(range(1,n_kf)):
        f_pre = result_keyframe[i_kf-1]['frame']
        f_crnt = result_keyframe[i_kf]['frame']

        pre_ids = copy.deepcopy(crnt_ids)

        for i_box, pid in enumerate(pre_ids):            
            if pid not in Trk.keys():
                Trk[pid] = -np.ones([n_frame, n_cam], dtype=int)            
            Trk[pid][f_pre:f_crnt,:] = result_keyframe[i_kf-1]['bcomb'][i_box]

        c = result_keyframe_connection[i_kf-1]

        crnt_ids = -np.ones(len(result_keyframe[i_kf]['bcomb']), dtype=int)
        for i_c in range(len(c)):
            crnt_ids[c[i_c][1]] = pre_ids[c[i_c][0]]

        for i_ids in range(len(crnt_ids)):
            if crnt_ids[i_ids] < 0:
                crnt_ids[i_ids] = cnt
                cnt += 1

    ### get cid info for each frames in each trace
    ####classnames = ['B', 'd','G', 'R','unknown', 'W']
    classnames = ['B', 'G', 'R', 'Y']
    n_class = len(classnames)

    conf_thr = 0.9#0.8
    conf_thr = 0.96#0.8 # 2022/8/4 for v0p324

    Trk_cid = {}

    for k in tqdm(Trk.keys()):
        trk = Trk[k]

        t_cid = np.zeros([n_frame, n_class], dtype=int)

        for i_cam in range(n_cam):
            boxid = trk[:,i_cam]

            for i_frame in range(n_frame):
                TT = T[i_cam][i_frame]
                for tt in TT:
                    if boxid[i_frame] == tt[0]:
                        cid = tt[6:]
                        if cid[1] > conf_thr:
                            t_cid[i_frame, int(cid[0])] += 1

        #Trk_cid[k] = t_cid[:,[0,2,3,5]]
        Trk_cid[k] = t_cid

    ### set cid for each frames
    Cid = {}
    fps = 24
    wsize = fps*5
    for k in tqdm(Trk_cid.keys()):
        cid0 = Trk_cid[k]
        # initialize labels
        cnt = np.sum(cid0, axis=0)
        i_max = np.argmax(cnt)
        if np.sum(cnt) == 0:
            p = 0.0
        else:
            p = cnt[i_max] / np.sum(cnt)

        if p > 0.8 and cnt[i_max] > 12:
            init_id = i_max
        else:
            init_id = -1
        cid = np.ones(n_frame, dtype=int) * init_id

        for i_frame in range(1, n_frame-wsize):
            if np.sum(cid0[i_frame,:]) == 0:
                continue
            if np.argmax(cid0[i_frame,:]) == cid[i_frame]:
                continue
            
            cnt = np.sum(cid0[i_frame:i_frame+wsize,:], axis=0)
            i_max = np.argmax(cnt)
            if np.sum(cnt) == 0:
                p = 0.0
            else:
                p = cnt[i_max] / np.sum(cnt)

            if i_max == cid[i_frame]:
                continue

            if p > 0.8 and cnt[i_max] > 12:
                cid[i_frame:-1] = i_max

        Cid[k] = cid

    ### clearn duplicated IDs
    for i_frame in tqdm(range(n_frame)):
        cid = []
        for k in Trk_cid.keys():
            if np.sum(Trk[k][i_frame]>=0) == 0:
                cid.append(-1)
            else:
                cid.append(Cid[k][i_frame])

        cid = np.array(cid)
        keys = list(Trk_cid.keys())

        for i_sub in range(4):
            if np.sum(cid==i_sub) > 1:
                I = np.argwhere(cid==i_sub).ravel()

                cnt = np.zeros(I.shape[0], dtype=int)

                for i_iter in range(4):
                    for ii, i in enumerate(I):
                        cnt[ii] = np.sum(Trk_cid[keys[i]][i_frame:min(i_frame+wsize*(i_iter+1), n_frame),i_sub])
                    if np.sum(cnt) != 0:
                        break

                for ii, i in enumerate(I):
                    Cid[keys[i]][i_frame] = -1
                
                i = I[np.argmax(cnt)]
                Cid[keys[i]][i_frame] = i_sub

    # divide tracklets at ID shifts
    Trk, Cid = div_3dtracklet(Trk, Cid)

    # backup before post-process
    with open(result_dir + '/_track.pickle', 'wb') as f:
        pickle.dump(Trk, f)
    with open(result_dir + '/_collar_id.pickle', 'wb') as f:
        pickle.dump(Cid, f)

    # triangulation
    camparam = get_camparam(config_path)
    n_kp = 18
    P3d = {}

    for k in tqdm(Trk.keys()):

        trk = Trk[k]

        n_frame, n_cam = trk.shape

        p3d = np.zeros([n_frame,n_kp,3], dtype=float)
        p3d[:,:,:] = np.nan

        for i_frame in range(n_frame):

            if np.sum(trk[i_frame]>=0) < 2:
                continue

            p2d = np.zeros([n_cam, n_kp, 3])
            p2d[:,:,:] = np.nan

            for i_cam in range(n_cam):
                TT = T[i_cam][i_frame]
                for tt in TT:
                    bbox_id = tt[0]
                    kp = np.array(tt[5])

                    if bbox_id == trk[i_frame][i_cam]:
                        p2d[i_cam, :,:] = kp

            p3d[i_frame, :, :] = calc_3dpose(p2d, config_path,camparam)

        P3d[k] = np.nanmean(p3d, axis=1)

    # dilate tracks (for two times)
    Cid_conflevel = {}
    for k in Trk.keys():
        if np.sum(Cid[k]>=0) == 0:
            Cid_conflevel[k] = -1   # unassigned
        else:
            Cid_conflevel[k] = 1   # assigned

    Trk, Cid, Cid_conflevel = dilate_track(Trk, Cid, Cid_conflevel, P3d, conflevel=2, rmse_thr=rmse_thr)
    Trk, Cid, Cid_conflevel = dilate_track(Trk, Cid, Cid_conflevel, P3d, conflevel=3, rmse_thr=rmse_thr)

    Trk, Cid, Cid_conflevel = assign_lastone(Trk, Cid, Cid_conflevel, P3d)

    with open(result_dir + '/track.pickle', 'wb') as f:
        pickle.dump(Trk, f)
    with open(result_dir + '/collar_id.pickle', 'wb') as f:
        pickle.dump(Cid, f)
    with open(result_dir + '/collar_id_conflv.pickle', 'wb') as f:
        pickle.dump(Cid_conflevel, f)

def visualize(vis_cam, config_path, result_dir, raw_data_dir,redo=False):

    camparam = get_camparam(config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    
    if os.path.exists('{:d}.mp4'.format(ID[vis_cam])) & (not redo):
        return 

    T = []
    for i_cam, id in enumerate(ID):
        with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
            data = json.load(f)
        T.append(data)

    data_name = os.path.basename(result_dir)
    n_cam = len(ID)
    S = []
    F = []
    for i_cam in range(n_cam):
        mdata = raw_data_dir + '/' + data_name + '.' + str(ID[i_cam]) + '/metadata.yaml'
        frame_num = np.load(result_dir + '/' + str(ID[i_cam]) + '/frame_num.npy')
        store = imgstore.new_for_filename(mdata)
        S.append(store)
        F.append(frame_num)

    with open(result_dir + '/track.pickle', 'rb') as f:
        Trk = pickle.load(f)
    with open(result_dir + '/collar_id.pickle', 'rb') as f:
        Cid = pickle.load(f)

    n_frame = Trk[list(Trk.keys())[0]].shape[0]

    n_kp = 18

    vw = cv2.VideoWriter('{:d}.mp4'.format(ID[vis_cam]), cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 480))
    #vw = cv2.VideoWriter('{:d}.mp4'.format(ID[vis_cam]), cv2.VideoWriter_fourcc(*'mp4v'), 24, (2048, 1536))

    for i_frame in tqdm(range(n_frame)):
    #for i_frame in tqdm(range(200)):

        i = F[vis_cam][i_frame]
        img, (frame_number, frame_time) = S[vis_cam].get_image(frame_number=i)

        for k in Trk.keys():
            trk = Trk[k][i_frame,:]
            if np.sum(trk>=0) == 0:
                continue
            p2d = np.zeros([n_cam, n_kp, 3])
            p2d[:,:,:] = np.nan

            for i_cam in range(n_cam):
                TT = T[i_cam][i_frame]
                for tt in TT:
                    bbox_id = tt[0]
                    kp = np.array(tt[5])

                    if bbox_id == trk[i_cam]:
                        p2d[i_cam, :,:] = kp
        
            p3d = calc_3dpose(p2d, config_path,camparam)
                
            clrs = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
            if Cid[k][i_frame] >= 0:
                clr = clrs[Cid[k][i_frame]]
            else:
                clr = (0, 0, 0)

            x = p3d
            a = (x[5,:] + x[6,:])/2
            x = np.concatenate([x,a[np.newaxis,:]],axis=0)
            p = reproject(vis_cam, x)
            p = np.concatenate([p, np.ones([p.shape[0], 1])], axis=1)
            kp = p.tolist()
            mrksize = 3
            clean_kp(kp)
            draw_kps(img, kp, mrksize, clr)

            x_min = np.nanmin(p[:,0])
            y_min = np.nanmin(p[:,1])

            if x_min > 3000 or x_min < -1000:
                x_min = np.nan
            if y_min > 3000 or y_min < -1000:
                y_min = np.nan

            if np.isnan(x_min) != True and np.isnan(y_min) != True:
                cv2.putText(img, str(k), (int(x_min), int(y_min)), #(int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2.0,
                    color=clr, thickness=5)
        
        img = cv2.resize(img, [640,480])
        vw.write(img)

    vw.release()

def create_kp2dfile(config_path, result_dir):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    T = []
    for i_cam, id in enumerate(ID):
        with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
            data = json.load(f)
        T.append(data)

    with open(result_dir + '/track.pickle', 'rb') as f:
        Trk = pickle.load(f)
    with open(result_dir + '/collar_id.pickle', 'rb') as f:
        Cid = pickle.load(f)
    with open(result_dir + '/collar_id_conflv.pickle', 'rb') as f:
        Cid_conflevel = pickle.load(f)

    n_frame = Trk[list(Trk.keys())[0]].shape[0]

    n_animal = 4
    n_cam = 8
    n_kp = 18

    kp2d = np.zeros([n_animal, n_frame, n_cam, n_kp, 3])
    conflv = np.ones([n_animal, n_frame, n_cam]) * 1000

    for i_frame in tqdm(range(n_frame)):

        for k in Trk.keys():

            i_animal = Cid[k][i_frame]

            if i_animal < 0:
                continue

            trk = Trk[k][i_frame,:]
            if np.sum(trk>=0) == 0:
                continue

            #p2d = np.zeros([n_cam, n_kp, 3])

            for i_cam in range(n_cam):
                if conflv[i_animal, i_frame, i_cam] < Cid_conflevel[k]:
                    continue

                TT = T[i_cam][i_frame]
                for tt in TT:
                    bbox_id = tt[0]
                    kp = np.array(tt[5])

                    if bbox_id == trk[i_cam]:
                        
                        kp2d[i_animal, i_frame, i_cam, :, :] = kp
                        conflv[i_animal, i_frame, i_cam] = Cid_conflevel[k]
        
            #kp2d[Cid[k][i_frame], i_frame, :, :, :] = p2d

    with open(result_dir + '/kp2d.pickle', 'wb') as f:
        pickle.dump(kp2d, f)

def proc(data_name,result_dir_root,raw_data_dir,config_path,save_vid=False,save_vid_cam=2,rmse_thr = 50.0,redo=False):
    result_dir = result_dir_root + '/'+data_name
    
    if redo | \
        (not os.path.exists(result_dir + '/keyframe_connection.pickle')):
        connect_keyframe(config_path, result_dir)
  
    if  redo | \
        (not os.path.exists(result_dir + '/track.pickle'))| \
        (not os.path.exists(result_dir + '/collar_id.pickle'))| \
        (not os.path.exists(result_dir + '/collar_id_conflv.pickle')): 
        analyze_cid(config_path, result_dir, rmse_thr=rmse_thr)
    if save_vid:      
        visualize(save_vid_cam, config_path, result_dir, raw_data_dir)
    
    if  redo | \
        (not os.path.exists(result_dir + '/kp2d.pickle')):
        create_kp2dfile(config_path, result_dir)
    
if __name__ == '__main__':

    config_path = './calib/marmo/config.yaml'
    data_name = 'pairisolation_cj542_cj835_20211222_161602'
    result_dir_root = './results/3d_RBIv0p1'
    raw_data_dir = './raw_data'

    config_path = './calib/marmo/config.yaml'
    data_name = 'tmp_demosession'
    result_dir_root = './tmptest'
    raw_data_dir = './raw_data'


    rmse_thr = 50.0 # in millimeter, to connect unassigned tracklets  # 2022/8/4 変更不要；この閾値は過去に変数化済み（マカク：200）

    save_vid = True#False
    save_vid_cam = 2

    proc(data_name,result_dir_root,raw_data_dir,config_path,save_vid,save_vid_cam,rmse_thr)   