# coding: utf-8

# MIT License
# 
# Copyright (c) 2018 Duong Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
A script to run the task-specific blocks of GeoTrackNet.
The code is adapted from
https://github.com/tensorflow/models/tree/master/research/fivo

"""

import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
import pickle
from tqdm import tqdm
import logging
import math
import scipy.special
from scipy import stats
from tqdm import tqdm
import csv
from datetime import datetime
import utils
import contrario_utils


import runners
from flags_config import config
# flags_configx中使用了，tf.app.flags用于处理命令行参数。
# 在flags_config中已经设定了许多的预设参数


# 下面是经度维度的范围，会根据config中的lat_max/min和lon_max/min自动计算，暂时不用在geotracknet.py中修改
LAT_RANGE = config.lat_max - config.lat_min
LON_RANGE = config.lon_max - config.lon_min

# 分辨率设定
FIG_DPI = 150
# W是宽度，H是高度
FIG_W = 960
FIG_H = int(FIG_W*LAT_RANGE/LON_RANGE)

# 概率模型参数设置，std、mean的阈值设定
LOGPROB_MEAN_MIN = -10.0
LOGPROB_STD_MAX = 5

# done

## RUN TRAIN
#======================================

if config.mode == "train":
    print(config.trainingset_path,"trainsetpath")
    fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
    tf.logging.set_verbosity(tf.logging.INFO)
    # get TF logger
    logger = logging.getLogger('tensorflow')
    logger.addHandler(fh)
    # runners 执行训练步骤
    runners.run_train(config)

else:
    with open(config.testset_path,"rb") as f:
        Vs_test = pickle.load(f)
    dataset_size = len(Vs_test)
    print("dataset size:")
    print(dataset_size)
    print("------------------")
    
## RUN TASK-SPECIFIC SUBMODEL
#======================================
# 该部分主要设置了与学习模型相关的参数
step = None
if config.mode in ["save_logprob","traj_reconstruction"]:
    tf.Graph().as_default()
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, bnum, time_starts, time_ends, lengths, model = runners.create_dataset_and_model(config,
                                                               shuffle=False,
                                                               repeat=False)

    if config.mode == "traj_reconstruction":
        config.missing_data = True
    #else:
    #    config.missing_data = False

    track_sample, track_true, log_weights, ll_per_t, ll_acc,_,_,_\
                                        = runners.create_eval_graph(inputs, targets,
                                                               lengths, model, config)
    saver = tf.train.Saver()
    sess = tf.train.SingularMonitoredSession()
    runners.wait_for_checkpoint(saver, sess, config.logdir)
    step = sess.run(global_step)

#runners.wait_for_checkpoint(saver, sess, config.logdir)
#step = sess.run(global_step)
#print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
## done


if step is None:
    # The log filename contains the step.
    index_filename = sorted(glob.glob(config.logdir+"/*.index"))[-1] # the lastest step
    step = int(index_filename.split(".index")[0].split("ckpt-")[-1])
    

print("Global step: ", step)
outputs_path = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "logprob-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name) + "-"\
            + str(config.latent_size)\
            + "-missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            + ".pkl"
if not os.path.exists(os.path.dirname(outputs_path)):
    os.makedirs(os.path.dirname(outputs_path))

save_dir = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "local_logprob-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name).replace("test","valid") + "-"\
            + str(config.latent_size) + "-"\
            + "missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            +"/"      

#===============================================================================
#===============================================================================
if config.mode == "save_logprob":
    """ save_logprob
    Calculate and save log[p(x_t|h_t)] of each track in the test set.
    """
    l_dict = []
    for d_i in tqdm(list(range(math.ceil(dataset_size/config.batch_size)))):
        inp, tar, bnum, t_start, t_end, seq_len, log_weights_np, true_np, ll_t =\
                 sess.run([inputs, targets, bnum, time_starts, time_ends, lengths, log_weights, track_true, ll_per_t])
        for d_idx_inbatch in range(inp.shape[1]):
            print("------")
            D = dict()
            seq_len_d = seq_len[d_idx_inbatch]
            # nonzero_indices = np.nonzero(tar[:seq_len_d,d_idx_inbatch,:])[1]
            # # 确保非零元素数量是5的倍数
            # num_elements = len(nonzero_indices)
            # num_to_remove = num_elements % 5
            # if num_to_remove != 0:
            #     nonzero_indices = nonzero_indices[:-num_to_remove]

            # # 重塑数组
            # reshaped_array = nonzero_indices.reshape(-1, 5)
            # print(reshaped_array)
            # # 然后将 reshaped_array 存入字典 D 的 "seq" 键中
            D["seq"] = np.nonzero(tar[:seq_len_d,d_idx_inbatch,:])[1].reshape(-1,5)
            D["t_start"] = t_start[d_idx_inbatch]
            D["t_end"] = t_end[d_idx_inbatch]
            D["bnum"] = bnum[d_idx_inbatch]
            D["log_weights"] = log_weights_np[:seq_len_d,:,d_idx_inbatch]
            #
            # print(D["log_weights"])
            l_dict.append(D)
    with open(outputs_path,"wb") as f:
        pickle.dump(l_dict,f)
        
    print(outputs_path)

    """ LL
    Plot the distribution of log[p(x_t|h_t)] of each track in the test set.
    """

    v_logprob = np.empty((0,))
    v_logprob_stable = np.empty((0,))

    count = 0
    for D in tqdm(l_dict):
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        v_logprob = np.concatenate((v_logprob,[ll_t]))

    d_mean = np.mean(v_logprob)
    d_std = np.std(v_logprob)
    d_thresh = d_mean - 3*d_std

    plt.figure(figsize=(1920/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)
    plt.plot(v_logprob,'o')
    plt.title("Log likelihood " + os.path.basename(config.testset_name)\
              + ", mean = {0:02f}, std = {1:02f}, threshold = {2:02f}".format(d_mean, d_std, d_thresh))
    plt.plot([0,len(v_logprob)], [d_thresh, d_thresh],'r')

    plt.xlim([0,len(v_logprob)])
    fig_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/" \
            + "logprob-" \
            + config.bound + "-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name)\
            + "-latent_size-" + str(config.latent_size)\
            + "-ll_thresh" + str(round(d_thresh, 2))\
            + "-missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            + ".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()
    
#===============================================================================
# 调试程序
#===============================================================================


#===============================================================================
#===============================================================================
elif config.mode == "contrario_detection":
    """ CONTRARIO DETECTION
    Detect abnormal vessels' behavior using a contrario detection.
    An AIS message is considered as abnormal if it does not follow the learned 
    distribution. An AIS track is considered as abnormal if many of its messages
    are abnormal.
    """      
    
    # Loading the parameters of the distribution in each cell (calculated by the
    # tracks in the validation set)
    with open(os.path.join(save_dir,"Map_logprob-"+\
              str(config.cell_lat_reso)+"-"+str(config.cell_lat_reso)+".pkl"),"rb") as f:
        Map_logprob = pickle.load(f)
    # Load the logprob
    with open(outputs_path,"rb") as f:
        l_dict = pickle.load(f)
    print("outputpath：",outputs_path)
    d_i = 0
    v_mean_log = []
    l_v_A = []
    v_buffer_count = []
    length_track = len(l_dict[0]["seq"])
    l_dict_anomaly = []
    n_error = 0
    for D in tqdm(l_dict):
        try:
        # if True:
            tmp = D["seq"]
            m_log_weights_np = D["log_weights"]
            # print("m_log_weights_np",m_log_weights_np)
            v_A = np.zeros(len(tmp))
            # print(tmp)
            for d_timestep in range(2*6,len(tmp)):
                # print(d_timestep)
                # print(config.cell_lat_reso,config.cell_lon_reso)
                d_row = int((tmp[d_timestep,4]-config.onehot_height_bins-config.onehot_speed_bins-config.onehot_angle_bins-config.onehot_lon_bins)*(8.0/300.0)/config.cell_lat_reso)
                d_col = int((tmp[d_timestep,3]-config.onehot_height_bins-config.onehot_speed_bins-config.onehot_angle_bins)*(22.0/300.0)/config.cell_lon_reso)
                # d_row = int(tmp[d_timestep,0]*config.onehot_lat_reso/config.cell_lat_reso)
                # d_col = int((tmp[d_timestep,1]-config.onehot_lat_bins)*config.onehot_lon_reso/config.cell_lon_reso)
                d_logprob_t = np.mean(m_log_weights_np[d_timestep,:])
                # print(d_row,d_col)
                # KDE
                l_local_log_prod = Map_logprob[str(d_row)+","+str(d_col)]
                if len(l_local_log_prod) < 2:
                    v_A[d_timestep] = 2
                else:
                    kernel = stats.gaussian_kde(l_local_log_prod)
                    cdf = kernel.integrate_box_1d(-np.inf,d_logprob_t)
                    if cdf < 0.05:
                        v_A[d_timestep] = 1
            v_A = v_A[12:]
            v_anomalies = np.zeros(len(v_A))
            for d_i_4h in range(0,len(v_A)+1-60):
                v_A_4h = v_A[d_i_4h:d_i_4h+60]
                v_anomalies_i = contrario_utils.contrario_detection(v_A_4h,config.contrario_eps)
                v_anomalies[d_i_4h:d_i_4h+60][v_anomalies_i==1] = 1

            if len(contrario_utils.nonzero_segments(v_anomalies)) > 0:
                D["anomaly_idx"] = v_anomalies
                l_dict_anomaly.append(D)
        except Exception as e:
            print("there is an error:",e)
            n_error += 1
    print("Number of processed tracks: ",len(l_dict))
    print("Number of abnormal tracks: ",len(l_dict_anomaly)) 
    print("Number of errors: ",n_error)
    
    # Save to disk
    n_anomalies = len(l_dict_anomaly)
    save_filename = os.path.basename(config.trainingset_name)\
                    +"-" + os.path.basename(config.trainingset_name)\
                    +"-" + str(config.latent_size)\
                    +"-missing_data-"+str(config.missing_data)\
                    +"-step-"+str(step)\
                    +".pkl"
    save_pkl_filename = os.path.join(save_dir,"List_abnormal_tracks-"+save_filename)
    with open(save_pkl_filename,"wb") as f:
        pickle.dump(l_dict_anomaly,f)
    
    ## Plot
    with open(config.trainingset_path,"rb") as f:
        Vs_train = pickle.load(f)
    with open(config.testset_path,"rb") as f:
        Vs_test = pickle.load(f)

    save_filename = "Abnormal_tracks"\
                + "-" + os.path.basename(config.trainingset_name)\
                + "-" + os.path.basename(config.testset_name)\
                + "-latent_size-" + str(config.latent_size)\
                + "-step-"+str(step)\
                + "-eps-"+str(config.contrario_eps)\
                + "-" + str(n_anomalies)\
                + ".png"
    
    # Plot abnormal tracks with the tracks in the training set as the background
    utils.plot_abnormal_tracks(Vs_train,l_dict_anomaly,
                     os.path.join(save_dir,save_filename),
                     config.lat_min,config.lat_max,config.lon_min,config.lon_max,
                     config.onehot_height_bins,config.onehot_speed_bins,
                     config.onehot_angle_bins,
                     config.onehot_lat_bins,config.onehot_lon_bins,
                     background_cmap = "Blues",
                     fig_w = FIG_W, fig_h = FIG_H,
                    )
    plt.close()
    # Plot abnormal tracks with the tracks in the test set as the background
    utils.plot_abnormal_tracks(Vs_test,l_dict_anomaly,
                     os.path.join(save_dir,save_filename.replace("Abnormal_tracks","Abnormal_tracks2")),
                     config.lat_min,config.lat_max,config.lon_min,config.lon_max,
                     config.onehot_height_bins,config.onehot_speed_bins,
                     config.onehot_angle_bins,
                     config.onehot_lat_bins,config.onehot_lon_bins,
                     background_cmap = "Greens",
                     fig_w = FIG_W, fig_h = FIG_H,
                    )
    plt.close()   
    # Save abnormal tracks to csv file
    with open(os.path.join(save_dir,save_filename.replace(".png",".csv")),"w") as f:
        writer = csv.writer(f)
        writer.writerow(["BNUM","Time_start","Time_end","Timestamp_start","Timestamp_end"])
        for D in l_dict_anomaly:
            writer.writerow([D["bnum"],
                             datetime.utcfromtimestamp(D["t_start"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                             datetime.utcfromtimestamp(D["t_end"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                             D["t_start"],D["t_end"]])
