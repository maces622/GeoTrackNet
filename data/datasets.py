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
Input pipelines script for Tensorflow graph.
This script is adapted from the original script of FIVO.
"""


from turtle import speed
import numpy as np
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('..')
import os
import pickle
import tensorflow as tf


#使每个AIS变量名称对应一个整数索引，从0开始。
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12


# In[]

def create_AIS_dataset(dataset_path,
                       mean_path,
                       batch_size,
                       data_dim,
                       lat_bins,
                       lon_bins,
                       sog_bins,
                       cog_bins,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True):
    total_bins = lat_bins+lon_bins+sog_bins+cog_bins
    def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis, time_start, time_end):
        # 实现论文中提到的four-hot编码
        #lat_bins = 200; lon_bins = 300; sog_bins = 30; cog_bins = 72
        def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
            lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
            data_dim = lat_bins + lon_bins + sog_bins + cog_bins
            dense_vect = np.zeros(data_dim)
            dense_vect[int(lat*lat_bins)] = 1.0
            dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
            dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
            dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0
            return dense_vect
        msgs_[msgs_ == 1] = 0.99999
        dense_msgs = []
        for msg in msgs_:
            # lat_bins, lon_bins, sog_bins, cog_bins are from "create_AIS_dataset" scope 
            dense_msgs.append(create_dense_vect(msg,
                                                lat_bins = lat_bins,
                                                lon_bins = lon_bins,
                                                sog_bins = sog_bins,
                                                cog_bins = cog_bins))
        dense_msgs = np.array(dense_msgs)
        return dense_msgs, num_timesteps, mmsis, time_start, time_end


    # Load the data from disk.
    with tf.gfile.Open(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    num_examples = len(raw_data)
    dirname = os.path.dirname(dataset_path)

    with open(mean_path,"rb") as f:
        mean = pickle.load(f)

    def aistrack_generator():
        for k in list(raw_data.keys()):
            # ::2表示间隔一条数据进行选取。该语句后是选出需要的四个字段。
            tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), raw_data[k][0,MMSI], raw_data[k][0,TIMESTAMP], raw_data[k][-1,TIMESTAMP]
            ## tmp是位置有关数据；长度；MMSI识别码；开始时间戳；结束时间戳。
    dataset = tf.data.Dataset.from_generator(
                              aistrack_generator,
                              output_types=(tf.float64, tf.int64, tf.int64, tf.float32, tf.float32))
            
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)

    # Batch sequences togther, padding them to a common length in time.
    #用于批量创建数据集不同长度的数据会被填充
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None, total_bins ], [], [], [], [])
                                  )
    


    def process_AIS_batch(data, lengths, mmsis, time_start, time_end):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.to_float(tf.transpose(data, perm=[1, 0, 2]))
        lengths = tf.to_int32(lengths)
        mmsis = tf.to_int32(mmsis)
        targets = data

        # Mean center the inputs. 减去均值使数据更稳定
        inputs = data - tf.constant(mean, dtype=tf.float32,
                                    shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.用于处理不同长度的序列
        inputs *= tf.expand_dims(tf.transpose(
            tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths, mmsis, time_start, time_end

    dataset = dataset.map(process_AIS_batch,
                          num_parallel_calls=num_parallel_calls)


#    dataset = dataset.prefetch(num_examples)
    # 预取
    dataset = dataset.prefetch(50)

    #创建单次迭代器用于遍历其中的元素
    itr = dataset.make_one_shot_iterator()

    #从迭代器中取出下一个元素
    inputs, targets, lengths, mmsis, time_starts, time_ends = itr.get_next()
        
    return inputs, targets, mmsis, time_starts, time_ends, lengths, tf.constant(mean, dtype=tf.float32)

"""
create datasets for ADB-S datasets
"""

TIMESTAMP,BNUM,HEIGHT,SPEED,ANGLE,LON,LAT=list(range(6))

def create_ADB_dataset(dataset_path,
                       mean_path,
                       batch_size,
                       data_dim,
                       lat_bins,
                       lon_bins,
                       height_bins,
                       speed_bins,
                       angle_bins,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True):
    total_bins = lat_bins+lon_bins+height_bins+speed_bins+angle_bins
    def sparse_AIS_to_dense(msgs_,num_timesteps, bnum, time_start, time_end):
        # 实现论文中提到的four-hot编码
        #lat_bins = 200; lon_bins = 300; speed_bins = 30; angle_bins = 72
        def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, height_bins = 300 ,speed_bins = 100,angle_bins=72):
            lat,lon,hgt,spd,agl=msg[0],msg[1],msg[2],msg[3],msg[4]
            data_dim = lat_bins + lon_bins + height_bins + speed_bins+angle_bins
            dense_vect = np.zeros(data_dim)
            dense_vect[int(lat*lat_bins)] = 1.0
            dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
            dense_vect[int(hgt*height_bins) + lat_bins + lon_bins] = 1.0
            dense_vect[int(spd*speed_bins) + lat_bins + lon_bins + height_bins] = 1.0
            dense_vect[int(agl*angle_bins) + lat_bins + lon_bins + height_bins+speed_bins] = 1.0
            return dense_vect
        msgs_[msgs_ == 1] = 0.99999
        dense_msgs = []
        for msg in msgs_:
            # lat_bins, lon_bins, height_bins, speed_bins, angle_bins are from "create_AIS_dataset" scope 
            dense_msgs.append(create_dense_vect(msg,
                                                lat_bins = lat_bins,
                                                lon_bins = lon_bins,
                                                height_bins=height_bins,
                                                speed_bins= speed_bins,
                                                angle_bins=angle_bins))
        dense_msgs = np.array(dense_msgs)
        return dense_msgs, num_timesteps, bnum, time_start, time_end


    # Load the data from disk.
    with tf.gfile.Open(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    num_examples = len(raw_data)
    dirname = os.path.dirname(dataset_path)

    with open(mean_path,"rb") as f:
        mean = pickle.load(f)

    def adbtrack_generator():
        for k in list(raw_data.keys()):
            # ::2表示间隔一条数据进行选取。该语句后是选出需要的四个字段。
            # TIMESTAMP,BNUM,HEIGHT,ANGLE,LON,LAT=list(range(6))
            # tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp = raw_data[k][::,[HEIGHT,SPEED,ANGLE,LON,LAT]] #不间隔
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), raw_data[k][0,BNUM], raw_data[k][0,TIMESTAMP], raw_data[k][-1,TIMESTAMP]
            ## tmp是位置有关数据；长度；ADB-S使用bnum作为班次的识别码；开始时间戳；结束时间戳。
    dataset = tf.data.Dataset.from_generator(
                              adbtrack_generator,
                              output_types=(tf.float64, tf.int64, tf.int64, tf.float32, tf.float32))
            
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)

    # Batch sequences togther, padding them to a common length in time.
    #用于批量创建数据集不同长度的数据会被填充
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None, total_bins ], [], [], [], [])
                                  )
    


    def process_ADB_batch(data, lengths, bnum, time_start, time_end):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.to_float(tf.transpose(data, perm=[1, 0, 2]))
        lengths = tf.to_int32(lengths)
        bnum = tf.to_int32(bnum)
        targets = data

        # Mean center the inputs. 减去均值使数据更稳定
        inputs = data - tf.constant(mean, dtype=tf.float32,
                                    shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.用于处理不同长度的序列
        inputs *= tf.expand_dims(tf.transpose(
            tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths, bnum, time_start, time_end

    dataset = dataset.map(process_ADB_batch,
                          num_parallel_calls=num_parallel_calls)


#    dataset = dataset.prefetch(num_examples)
    # 预取
    dataset = dataset.prefetch(50)

    #创建单次迭代器用于遍历其中的元素
    itr = dataset.make_one_shot_iterator()

    #从迭代器中取出下一个元素
    inputs, targets, lengths, bnum, time_starts, time_ends = itr.get_next()
        
    return inputs, targets, bnum, time_starts, time_ends, lengths, tf.constant(mean, dtype=tf.float32)

## for ADB-S datasets
