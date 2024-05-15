# step b1
# get sync data of multicam recording session

import json
import imgstore
import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
import joblib
import argparse
import platform

def proc(data_name,results_dir_root, raw_data_dir, label2d_dir,fps,t_intv, redo=False):
    L = glob.glob(raw_data_dir + '/' + data_name + '.*/metadata.yaml')

    #print(L)
    stores = []
    for l in L:
        store = imgstore.new_for_filename(l)
        stores.append(store)

    def get_nearest(a, A):
        d = np.abs(A-a)
        return np.argmin(d)

    mdata = stores[0].get_frame_metadata()
    t = mdata['frame_time']

    if t_intv is None:
        t_start = t[0]+1
        t_end = t[-1]-1
    else:
        t_start = t[0]+t_intv[0]
        t_end = t[0]+t_intv[1]

    T = np.arange(t_start, t_end, 1/fps)

    def process_single_cam(i_store,redo):       
        store = stores[i_store]
        cname = os.path.basename(store.filename).split('.')[1]

        out_dir = results_dir_root+'/' + data_name + '/' + cname
        path_json_alldata = out_dir + '/alldata.json'
        path_npy_frame_num = out_dir + '/frame_num.npy'
        if os.path.exists(path_json_alldata) & os.path.exists(path_npy_frame_num) & (not redo):
            return 

        L = glob.glob(label2d_dir + '/' + data_name + '/' + data_name + '_' + cname + '_*.json')
        L.sort()
        #print(L)
        
        D = []

        last_boxid = 0
        for l in L:
            with open(l, 'r') as f:
                data = json.load(f)

            max_boxid = -1
            for i_frame, _ in enumerate(data):
                if len(data[i_frame]) > 0:
                    for i_box, _ in enumerate(data[i_frame]):
                        max_boxid = max(max_boxid, data[i_frame][i_box][0])
                        data[i_frame][i_box][0] += last_boxid

            last_boxid += max_boxid+1
            
            D = D + data

        mdata = store.get_frame_metadata()
        t_cam = mdata['frame_time']
        frame_num = mdata['frame_number']
        
        D2 = []
        F = []

        for t in tqdm(T):
            i = get_nearest(t, t_cam)
            D2.append(D[i])
            F.append(frame_num[i])
            
        F = np.array(F, dtype=int)

        #print(len(D), len(t_cam), len(D2), F.shape)    
        os.makedirs(out_dir, exist_ok=True)

        with open(path_json_alldata, 'w') as f:
            json.dump(D2, f)

        np.save(path_npy_frame_num, F)

    
    for i_store in range(len(stores)):
        process_single_cam(i_store,redo)
    # result = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(joblib.delayed(process_single_cam)(i_store) for i_store in range(len(stores)))

if __name__ == '__main__':

    if 0: # legacy of devel
        t_intv = [10, 300]# use None to analyze all
        data_name = 'dailylife_cj425_20220402_110000'
        # raw_data_dir = './video'
        if platform.system()=='Windows':
            raw_data_dir ='Y:/tkaneko/marmo2hoMotif/dailylife'
        else:
            raw_data_dir= '/mnt/amakusa4/DataOrg/tkaneko/marmo2hoMotif/dailylife'
        label2d_dir = './results'
        results_dir_root='./results3D'
        fps = 24
    else: 
        parser = argparse.ArgumentParser()
        parser.add_argument('--t_intv', default='10,300', type=str, help='time interval to analyze, e.g. "10, 300"')
        parser.add_argument('--data_name', default='', type=str, help='session name, e.g., dailylife_cj425_20220402_110000')
        parser.add_argument('--raw_data_dir', default='cuda:0', help='Root directory of motif videos')
        parser.add_argument('--label2d_dir', default='./results', help='Root directory of 2D keypoints')
        parser.add_argument('--results_dir_root', default='./results3D', help='Root directory to save 3D results')
        parser.add_argument('--fps', default=24, type=int, help='fsp')
        parser.add_argument('--redo', default=False, type=bool, help='redo and overwrite')
        args = parser.parse_args()
        t_intv=[0,0]
        t_intv[0]=int(args.t_intv.split(',')[0])
        t_intv[1]=int(args.t_intv.split(',')[1])
        data_name=args.data_name
        raw_data_dir=args.raw_data_dir
        label2d_dir=args.label2d_dir
        results_dir_root=args.results_dir_root
        fps=args.fps
        redo=args.redo 
    proc(data_name,results_dir_root, raw_data_dir, label2d_dir,fps,t_intv,redo)
