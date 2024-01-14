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
Distribution utils for MultitaskAIS. 
"""

#import numpy as np
from matplotlib.scale import LogitScale
from numpy import angle
import tensorflow as tf

def sample_one_hot(prob_):
    # 从输入的分布prob_采样
    prob = prob_/tf.tile(tf.reduce_sum(prob_,axis=1,keep_dims=True),
                               multiples=[1,tf.shape(prob_)[1]])
    cdv = tf.cumsum(prob,axis = 1, exclusive=False)
    cdv_ex = tf.cumsum(prob,axis = 1, exclusive=True)
    
    thresh = tf.tile(tf.random_uniform(shape=(tf.shape(prob)[0],1)),
                        multiples=[1,tf.shape(prob)[1]])
    
    less_equal = tf.less_equal(cdv_ex,thresh)
    greater = tf.greater(cdv,thresh)
    
    one_hot = tf.cast(tf.logical_and(less_equal,greater), tf.float32)
    return one_hot


"""sample_from_max_logits for ADB-S datasets"""
def sample_from_max_logits(logit,height_bins,speed_bins,angle_bins, lon_bins,lat_bins):
    # l_embedding_sizes = [lat_bins, lon_bins, sog_bins, cog_bins]
    l_embedding_sizes=[height_bins,speed_bins,angle_bins, lon_bins,lat_bins]
    logit_height, logit_height,logit_angle,logit_lon,logit_lat = tf.split(logit,l_embedding_sizes,axis=1)
    ind_hgt = tf.argmax(logit_height,axis = 1)
    ind_spd = tf.argmax(logit_height,axis = 1)
    ind_agl = tf.argmax(logit_angle,axis = 1)
    ind_lon = tf.argmax(logit_lon,axis = 1)
    ind_lat = tf.argmax(logit_lat,axis = 1)

    onehot_hgt = tf.one_hot(ind_hgt,height_bins)
    onehot_spd = tf.one_hot(ind_spd,speed_bins)
    onehot_agl = tf.one_hot(ind_agl,angle_bins)
    onehot_lon = tf.one_hot(ind_lon,lon_bins)
    onehot_lat = tf.one_hot(ind_lat,lat_bins)

    fourhot = tf.concat([onehot_hgt,onehot_spd,onehot_agl,onehot_lon,onehot_lat],axis = 1)
    return fourhot



"""sample from logits for ADB-S datasets"""
def sample_from_logits(logit,height_bins,speed_bins,angle_bins,lon_bins,lat_bins):
    l_embedding_sizes = [height_bins,speed_bins,angle_bins,lon_bins,lat_bins]
    
    logit_hgt,logit_spd,logit_alg,logit_lon,logit_lat = tf.split(logit,l_embedding_sizes,axis=1)
    dist_hgt = tf.contrib.distributions.Bernoulli(logits=logit_hgt)
    dist_spd = tf.contrib.distributions.Bernoulli(logits=logit_spd)
    dist_agl = tf.contrib.distributions.Bernoulli(logits=logit_alg)
    dist_lon = tf.contrib.distributions.Bernoulli(logits=logit_lon)
    dist_lat = tf.contrib.distributions.Bernoulli(logits=logit_lat)

    sample_hgt = dist_hgt.sample()
    sample_spd = dist_spd.sample()
    sample_agl = dist_agl.sample()
    sample_lon = dist_lon.sample()
    sample_lat = dist_lat.sample()

    sample_all = tf.concat([sample_hgt,sample_spd,sample_agl,sample_lon,sample_lat],axis = 1)
    return sample_all
    

"""sample_from_probs for ADB-S datasets"""
def sample_from_probs(probs_,height_bins,speed_bins,angle_bins,lon_bins,lat_bins):
    l_embedding_sizes = [height_bins,speed_bins,angle_bins,lon_bins, lat_bins]
    def squash_prob(l_old_prob):
        l_new_probs = []
        for old_prob in l_old_prob:
            new_probs0 = old_prob/tf.reshape(tf.reduce_max(old_prob,axis=1),(-1,1))
            new_probs1 = tf.where(tf.equal(new_probs0,1.), tf.ones(tf.shape(old_prob))*0.9999, new_probs0)
            l_new_probs.append(new_probs1)
        return l_new_probs

    prob_hgt,prob_spd,prob_agl,prob_lon, prob_lat = squash_prob(tf.split(probs_,l_embedding_sizes,axis=1))
    sample_hgt = sample_one_hot(prob_hgt)
    sample_spd = sample_one_hot(prob_spd)
    sample_agl = sample_one_hot(prob_agl)
    sample_lon = sample_one_hot(prob_lon)
    sample_lat = sample_one_hot(prob_lat)

    sample_all = tf.concat([sample_hgt,sample_spd,sample_agl,sample_lon,sample_lat],axis = 1)
    return sample_all



