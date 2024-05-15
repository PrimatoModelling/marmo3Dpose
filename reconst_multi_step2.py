# step b2
# find matching at keyframes

import os
import os.path as osp
import pickle
import sys
import time
import multicam_toolbox as mct
from pictorial import transform_closure
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
import argparse

model_cfg = {'joint_num': 18, 'spectral': True, 'alpha_SVT': 0.5,
             'lambda_SVT': 50,'dual_stochastic_SVT': False,}

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

def myproj2dpam(Y, tol=1e-4):
    X0 = Y
    X = Y
    I2 = 0

    for iter_ in range ( 10 ):

        X1 = projR ( X0 + I2 )
        I1 = X1 - (X0 + I2)
        X2 = projC ( X0 + I1 )
        I2 = X2 - (X0 + I1)

        chg = np.sum ( np.abs ( X2[:] - X[:] ) ) / X.numel ()
        X = X2
        if chg < tol:
            return X
    return X

def projR(X):
    for i in range ( X.shape[0] ):
        X[i, :] = proj2pav ( X[i, :] )
        # X[i, :] = proj2pavC ( X[i, :] )
    return X

def projC(X):
    for j in range ( X.shape[1] ):
        # X[:, j] = proj2pavC ( X[:, j] )
        # Change to tradition implementation
        X[:, j] = proj2pav ( X[:, j] )
    return X

def proj2pav(y):
    y[y < 0] = 0
    x = np.zeros_like ( y )
    if np.sum ( y ) < 1:
        x += y
    else:
        u, _ = np.sort ( y, descending=True )
        sv = np.cumsum ( u, 0 )
        to_find = u > (sv - 1) / (np.arange ( 1, len ( u ) + 1, device=u.device, dtype=u.dtype ))
        rho = np.nonzero ( to_find.reshape ( -1 ) )[-1]
        theta = np.maximum ( 0, (sv[rho] - 1) / (rho.float () + 1) )
        x += np.maximum ( y - theta, 0 )
    return x

def matchSVT(S, dimGroup, **kwargs):

    alpha = kwargs.get ( 'alpha', 0.1 )
    pSelect = kwargs.get ( 'pselect', 1 )
    tol = kwargs.get ( 'tol', 5e-4 )
    maxIter = kwargs.get ( 'maxIter', 500 )
    verbose = kwargs.get ( 'verbose', False )
    eigenvalues = kwargs.get ( 'eigenvalues', False )
    _lambda = kwargs.get ( '_lambda', 50 )
    mu = kwargs.get ( 'mu', 64 )
    dual_stochastic = kwargs.get ( 'dual_stochastic_SVT', True )
    if verbose:
        print ( 'Running SVT-Matching: alpha = %.2f, pSelect = %.2f _lambda = %.2f \n' % (
            alpha, pSelect, _lambda) )
    info = dict ()
    N = S.shape[0]
    S[np.arange ( N ), np.arange ( N )] = 0
    S = (S + S.T) / 2
    X = S.copy ()
    Y = np.zeros_like ( S )
    W = alpha - S
    t0 = time.time ()
    for iter_ in range ( maxIter ):

        X0 = X
        # update Q with SVT
        U, s, Vh = np.linalg.svd ( 1.0 / mu * Y + X , full_matrices=False)
        V = np.conjugate(Vh.T)
        diagS = s - _lambda / mu
        diagS[diagS < 0] = 0
        Q = U @ np.diag(diagS) @ V.T
        # update X
        X = Q - (W + Y) / mu
        # project X
        for i in range ( len ( dimGroup ) - 1 ):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        if pSelect == 1:
            X[np.arange ( N ), np.arange ( N )] = 1
        X[X < 0] = 0
        X[X > 1] = 1

        if dual_stochastic:
            # Projection for double stochastic constraint
            for i in range ( len ( dimGroup ) - 1 ):
                row_begin, row_end = int ( dimGroup[i] ), int ( dimGroup[i + 1] )
                for j in range ( len ( dimGroup ) - 1 ):
                    col_begin, col_end = int ( dimGroup[j] ), int ( dimGroup[j + 1] )
                    if row_end > row_begin and col_end > col_begin:
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam ( X[row_begin:row_end, col_begin:col_end],
                                                                                1e-2 )

        X = (X + X.T) / 2
        # update Y
        Y = Y + mu * (X - Q)
        # test if convergence
        pRes = np.linalg.norm ( X - Q ) / N
        dRes = mu * np.linalg.norm ( X - X0 ) / N
        if verbose:
            print ( f'Iter = {iter_}, Res = ({pRes}, {dRes}), mu = {mu}' )

        if pRes < tol and dRes < tol:
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = (X + X.T) / 2
    info['time'] = time.time () - t0
    info['iter'] = iter_

    if eigenvalues:
        info['eigenvalues'] = np.linalg.eig ( X )

    X_bin = X > 0.5
    if verbose:
        print ( f"Alg terminated. Time = {info['time']}, #Iter = {info['iter']}, Res = ({pRes}, {dRes}), mu = {mu} \n" )
    match_mat = transform_closure ( X_bin.astype(np.uint8))
    return match_mat

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

def undistort_points(config_path, i_cam, pos_2d):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    id = ID[i_cam]

    with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_intrin:
        K = f_intrin['/'+str(id)+'/K'][()]
        xi = f_intrin['/'+str(id)+'/xi'][()]
        D = f_intrin['/'+str(id)+'/D'][()]
    p = pos_2d+0.0
    p_undist = cv2.omnidir.undistortPoints(np.array([p.tolist()], np.float), K, D, xi, np.eye(3))
    p_undist = np.squeeze(p_undist)
    
    return p_undist

def deproject(config_path, i_cam, P2d, depth, camparam=None): 
    
    p_undist = P2d+0.0
    
    if p_undist.ndim == 1:
        p_undist = p_undist[np.newaxis, :]

    if camparam is None:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        ID = cfg['camera_id']
        id = ID[i_cam]
        
        with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', mode='r') as f_extrin:
            rvecs = f_extrin['/'+str(id)+'/rvec'][()]
            tvecs = f_extrin['/'+str(id)+'/tvec'][()]
            rmtx, jcb = cv2.Rodrigues(rvecs)
            R = np.hstack([rmtx, tvecs])
    else:
        ID = camparam['camera_id']
        tvecs = camparam['tvecs'][i_cam]
        rmtx = camparam['pmat'][i_cam][:3,:3]

    P3d_local = copy.deepcopy(p_undist)
    z = np.ones((p_undist.shape[0],1), float)

    P3d_local = np.concatenate([P3d_local,z], axis=1)
    P3d_local = P3d_local*depth

    R_inv = np.linalg.inv(rmtx)

    P3d_global = []
    for p in P3d_local:
        pp = p - tvecs.ravel()
        ppp = np.matmul(R_inv, pp.T)
        #ppp = R_inv * pp[:, np.newaxis]
        P3d_global.append(ppp.ravel())

    P3d_global = np.array(P3d_global)

    return P3d_global

def calc_dist_btw_lines(v1, v2):
    #https://risalc.info/src/distance-between-two-lines.html

    x1 = v1[0:3]
    x2 = v2[0:3]
    m1 = v1[3:6] - x1
    m2 = v2[3:6] - x2
    m1 = m1/np.linalg.norm(m1)
    m2 = m2/np.linalg.norm(m2)

    c = np.cross(m1,m2)

    d = np.abs(np.dot(x2-x1,c))/np.linalg.norm(c)

    return d

def geometry_affinity2(points_set, dimGroup, config_path,camparam=None):

    Dth2 = 150

    M, n_kp, _ = points_set.shape

    distance_matrix = np.ones ( (M, M), dtype=np.float32 ) * Dth2*2
    np.fill_diagonal(distance_matrix, 0)

    import time
    start_time = time.time()

    I_cam = []
    for i in range(M):
        I = np.argwhere(dimGroup>i)
        I_cam.append(np.min(I)-1)

    V = []
    for i_person in range(M):
        i_cam = I_cam[i_person]
        v1 = deproject(config_path, i_cam, points_set[i_person, :, :2], 0.0, camparam=camparam)
        v2 = deproject(config_path, i_cam, points_set[i_person, :, :2], 1000.0, camparam=camparam)
        v = np.concatenate([v1, v2], axis=1)
        V.append(v)

    S = []
    for i_person in range(M):
        S.append(points_set[i_person, :, 2].squeeze())

    for i in range(M):
        for j in range(i+1, M):
            if I_cam[i] == I_cam[j]:
                continue
            
            d = np.zeros(n_kp, dtype=float)
            for i_kp in range(n_kp):
                v_i = np.squeeze(V[i][i_kp,:])
                v_j = np.squeeze(V[j][i_kp,:])
                d[i_kp] = calc_dist_btw_lines(v_i, v_j)

            th = 0.3
            s_i = S[i]
            s_j = S[j]
            I = (s_i>th) * (s_j>th)
            if np.sum(I) == 0:
                continue
            if np.sum(I) < 3:
                continue
            distance_matrix[i,j] = np.mean(d[I])
            distance_matrix[j,i] = np.mean(d[I])
    
    end_time = time.time()
    #print('using %fs' % (end_time - start_time))

    #plt.imshow(distance_matrix)
    #plt.colorbar()
    #plt.show()

    dm_mean = distance_matrix[distance_matrix<Dth2*2].mean()
    dm_std = distance_matrix[distance_matrix<Dth2*2].std()
    affinity_matrix = - (distance_matrix - dm_mean) / dm_std

    affinity_matrix = 1 / (1 + np.exp ( -5 * affinity_matrix ))
    
    affinity_matrix[distance_matrix>Dth2] = 0

    return affinity_matrix

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

class MultiEstimator ( object ):
    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.dataset = None

    def predict(self, fpath, show=False, plt_id=0, camparam=None, bcomb_prev=[]):
        
        with open(fpath, 'rb') as f:
            info_dict = pickle.load(f)

        return self.predict_data( info_dict, show=show, plt_id=plt_id, camparam=camparam, bcomb_prev=bcomb_prev )

    def predict_data(self, info_dict, show=False, plt_id=0, camparam=None, bcomb_prev=[]):

        n_cam = 8

        camnames = list(range(n_cam))

        dimGroup = [0]
        cnt = 0
        for i_cam in range(n_cam):
            cnt += len(info_dict[i_cam][0])
            dimGroup.append(cnt)

        dimGroup = np.array(dimGroup)

        return self._estimate3d ( 0, dimGroup, camnames, info_dict, show=show, plt_id=plt_id, camparam=camparam, bcomb_prev=bcomb_prev )

    def _estimate3d(self, img_id, dimGroup,  camnames, info_dict, show=False, plt_id=0, camparam=None, bcomb_prev=[]):

        n_cam = 8
        n_kp = 18

        info_list = list ()
        for cam_id in camnames:
            info_list += info_dict[cam_id][img_id]

        if len(info_list)==0:
            print('No detection!')
            return [], [], []

        pose_mat = np.array ( [i['pose2d'] for i in info_list] ).reshape ( -1, model_cfg['joint_num'], 2 )[..., :2]

        pose_score = np.array ( [i['pose2d_raw'] for i in info_list] ).reshape ( -1, model_cfg['joint_num'], 3 )[..., 2]

        pose_mat = np.concatenate([pose_mat, pose_score[:,:,np.newaxis]], axis=2)

        sub_imgid2cam = np.zeros ( pose_mat.shape[0], dtype=np.int32 )
        for idx, i in enumerate ( range ( len ( dimGroup ) - 1 ) ):
            sub_imgid2cam[dimGroup[i]:dimGroup[i + 1]] = idx

        det_box_list=[i['bbox'] for i in info_list]
        kpraw_list=[i['pose2d_raw'] for i in info_list]
        det_boxid_list=[i['bbox_id'] for i in info_list]

        geo_affinity_mat = geometry_affinity2( pose_mat.copy(), dimGroup, self.cfg, camparam=camparam)

        continuity_mat = np.ones(geo_affinity_mat.shape, dtype=np.float32) * 1.0
        for bc in bcomb_prev:
            I = []
            for i_cam in range(n_cam):
                for i_box, bid in enumerate(det_boxid_list):
                    if bid[0] == i_cam and bc[i_cam] == bid[1]:
                        I.append(i_box)
            #print(I)
            for i in I:
                for j in I:
                    continuity_mat[i,j] = 1.0       ##### CHANGE HERE FOR WEIGHT

        W = geo_affinity_mat  # Affinity matrix = Geometry constraint only

        W[np.isnan ( W )] = 0  # Some times (Shelf 452th img eg.) torch.sqrt will return nan if its too small

        num_person = 4
        X0 = np.random.rand ( W.shape[0], num_person )

        # Use spectral method to initialize assignment matrix.
        if self.cfg['spectral']:
            eig_value, eig_vector = np.linalg.eig(W)
            eig_idx = np.argsort (eig_value)[::-1]
            if W.shape[1] >= num_person:
                X0 = eig_vector[eig_idx[:num_person]].T
            else:
                X0[:, :W.shape[1]] = eig_vector.T


        match_mat = matchSVT ( W, dimGroup, alpha=self.cfg['alpha_SVT'], _lambda=self.cfg['lambda_SVT'],
                               dual_stochastic_SVT=self.cfg['dual_stochastic_SVT'])

        bin_match = match_mat[:, np.squeeze(np.nonzero( np.sum(match_mat, axis=0) > 1.9 ))] > 0.9
        bin_match = bin_match.reshape ( W.shape[0], -1 )

        matched_list = [[] for i in range ( bin_match.shape[1] )]
        for sub_imgid, row in enumerate ( bin_match ):
            if row.sum () != 0:
                pid = row.argmax ()
                matched_list[pid].append ( sub_imgid )

        matched_list = [np.array ( i ) for i in matched_list]

        def get_bestcomb(person, sub_imgid2cam, kpraw_list, n_cam):

            cam_ids = sub_imgid2cam[person]

            cam_groups = []
            for i_cam in range(n_cam):
                if np.sum(cam_ids==i_cam) == 0:
                    cam_groups.append([None])
                else:
                    cam_groups.append(person[cam_ids==i_cam].tolist())

            comb_list = list(itertools.product(cam_groups[0], cam_groups[1], cam_groups[2], 
                                               cam_groups[3], cam_groups[4], cam_groups[5],
                                               cam_groups[6], cam_groups[7]))

            if len(comb_list) == 1:
                return person

            E = []
            for i_comb, box_comb in enumerate(comb_list):
                kp2d = np.zeros([n_cam, n_kp, 3])
                for i_cam, i_box in enumerate(box_comb):
                    if i_box is None:
                        continue
                    kp2d[i_cam, :, :] = kpraw_list[i_box]

                pose3d = calc_3dpose(kp2d, self.cfg,camparam=camparam)

                D = []
                for i_cam in range(n_cam):
                    if box_comb[i_cam] is None:
                        continue
                    kp2d_reproj = reproject(i_cam, pose3d, camparam=camparam)
                    d = kp2d[i_cam, :, :2] - kp2d_reproj
                    d = d[kp2d[i_cam, :, 2] > 0.3, :]
                    D.append(d)
                D = np.concatenate(D, axis=0)
                rmse = np.sqrt(np.sum(D**2)/D.shape[0])
                E.append(rmse)

            E = np.array(E)
            i_best = np.argmin(E)

            ##print('combination error was fixed:', comb_list[i_best])
            I = np.array(comb_list[i_best])
            I = I[np.logical_not(I==None)].astype(int)
        
            return I

        for i_person, person in enumerate ( matched_list ):

            # list-up all possible combinations

            #print('---', i_person)

            #print(cam_ids, person)
            best_comb = get_bestcomb(person, sub_imgid2cam, kpraw_list, n_cam)

            ori_set = set(matched_list[i_person])
            sel_set = set(best_comb)
            matched_list[i_person] = best_comb
            if len(list(ori_set - sel_set)) > 1:
                rest = np.array(list(ori_set - sel_set), dtype=int)
                best_comb2 = get_bestcomb(rest, sub_imgid2cam, kpraw_list, n_cam)
                matched_list.append(best_comb2)
                #print('also added:', best_comb2)


        if show:

            clrs = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0)]
            for cam_id in camnames:
                img = info_dict[cam_id]['image_data']
                for i, person in enumerate ( matched_list ):
                    for sub_imageid in person:
                        cam_id2 = sub_imgid2cam[sub_imageid]
                        if cam_id2 == cam_id:
                            bb = np.array(det_box_list[sub_imageid][:4], dtype=int)
                            cv2.rectangle ( img, (bb[0], bb[1]), (bb[2], bb[3]), clrs[i], 5 )
                            cv2.putText(img, str(sub_imageid), (int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=3.0, color=(255, 255, 255), thickness=5, lineType=cv2.LINE_4)
                            kp = kpraw_list[sub_imageid].tolist()
                            mrksize = 3
                            clean_kp(kp)
                            draw_kps(img, kp, mrksize, (0,0,0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.show()

            print(matched_list)
            
            plt.imshow(W)
            plt.colorbar()
            plt.show()

        P3d = []
        matched_list2 = []
        Bcomb = []
        for mlist in matched_list:
            if mlist.shape[0] < 2:
                continue
            kp2d = np.zeros([n_cam, n_kp, 3])
            for i in mlist:
                i_cam = sub_imgid2cam[i]
                kp2d[i_cam, :, :] = kpraw_list[i]
            
            pose3d = calc_3dpose(kp2d,self.cfg, camparam=camparam)
            P3d.append(pose3d)
            
            bcomb = -np.ones(n_cam, dtype=int)
            for i in mlist:
                i_cam = sub_imgid2cam[i]
                bcomb[i_cam] = det_boxid_list[i][1]
            
            matched_list2.append(mlist)
            Bcomb.append(bcomb)
            
        return matched_list, P3d, Bcomb

def proc(data_name,result_dir_root,raw_data_dir,config_path,show_result=False,redo=False):
    result_dir = result_dir_root + '/' + data_name
    camparam = get_camparam(config_path)

    model_cfg = {'joint_num': 18, 'spectral': True, 'alpha_SVT': 0.5,
             'lambda_SVT': 50,'dual_stochastic_SVT': False,}
    test_model = MultiEstimator ( cfg=model_cfg )
        
    if os.path.exists(result_dir + '/match_keyframe.pickle') & (not redo):
        return 

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    T = []
    for i_cam, id in enumerate(ID):

        with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
            data = json.load(f)

        T.append(data)

    data_name = os.path.basename(result_dir)
    n_cam = len(ID)
    n_frame = len(T[0])

    S = []
    F = []
    for i_cam in range(n_cam):
        mdata = raw_data_dir + '/' + data_name + '.' + str(ID[i_cam]) + '/metadata.yaml'
        frame_num = np.load(result_dir + '/' + str(ID[i_cam]) + '/frame_num.npy')
        store = imgstore.new_for_filename(mdata)
        S.append(store)
        F.append(frame_num)

    result_keyframe = []
    bcomb_prev = []
    for i_frame in tqdm(range(1, n_frame-24*2, 12)):
    #for i_frame in tqdm(range(1, n_frame-24*5, 4)):

        info_dict = {}
        for i_cam in range(n_cam):

            if show_result:
                i = F[i_cam][i_frame]
                img, (frame_number, frame_time) = S[i_cam].get_image(frame_number=i)
            else:
                img = []

            TT = T[i_cam][i_frame]

            P = []
            for tt in TT:
                bbox_id = [i_cam, tt[0]]
                bbox = tt[1:5]
                pose2d_raw = np.array(tt[5])
                pose2d = undistort_points(config_path, i_cam, pose2d_raw[:,:2])

                P.append({'pose2d':pose2d, 'pose2d_raw':pose2d_raw, 'bbox':bbox, 'bbox_id':bbox_id})

            info_dict[i_cam] = {'image_data':img, 0:P}

        matched_list, pose3d, bcomb = test_model.predict_data ( info_dict, show=False, plt_id=0, camparam=camparam, bcomb_prev=bcomb_prev )

        bcomb_prev = bcomb

        if show_result:
            clrs = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0)]
            img = info_dict[vis_cam]['image_data']
            for i,p3d in enumerate(pose3d):
                x = p3d
                a = (x[5,:] + x[6,:])/2
                x = np.concatenate([x,a[np.newaxis,:]],axis=0)
                p = reproject(vis_cam, x)
                p = np.concatenate([p, np.ones([p.shape[0], 1])], axis=1)
                kp = p.tolist()
                mrksize = 3
                clean_kp(kp)
                draw_kps(img, kp, mrksize, clrs[i%7])
            img = cv2.resize(img, [640,480])
            #vw.write(img)
            cv2.imshow('test',img)
            cv2.waitKey(1)

        result_keyframe.append({'frame': i_frame, 'bcomb':bcomb, 'pose3d':pose3d})

    with open(result_dir + '/match_keyframe.pickle', 'wb') as f:
        pickle.dump(result_keyframe, f)

if __name__ == '__main__':

    #matplotlib.rcParams['figure.figsize'] = (7, 7)

    vis_cam = 5#2
    show_result = False#True

    data_name = 'collar_chd84_20220414_140000'
    raw_data_dir = './raw_data'
    config_path = './calib/config.yaml'
    result_dir_root='./results3D'   

    proc(data_name,result_dir_root,raw_data_dir,config_path)  
