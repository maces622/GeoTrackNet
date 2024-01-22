from lib2to3.pytree import BasePattern
import numpy as np
import os
import csv
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
# from .flags_config    import HEIGHT_MAX
# init the min\max value
lon_max=-90.0
lon_min=90.0
lat_max=-180
lat_min=180
hgt_max=0.0
hgt_min=100000.0
spd_max=0.0
spd_min=100000.0
agl_max=0.0
agl_min=360.0

data_set_name="CA1883"

file_cnt=0;
base_path="./CA_data/"
resp_path=data_set_name+"R"
save_path=data_set_name+"T"
csv_path=os.path.join(base_path,resp_path)
test_set_path=os.path.join(base_path,save_path)
data_dict={}
csv_cnt=0


# 在处理数据集的时候，提取出经纬度的最值，用于config_flags.py模块设定参数

# timestamp bnum height speed angle longitude latitude
bnum=10001
for fn in os.listdir(csv_path):
    csv_file_path=os.path.join(csv_path,fn)
    df=pd.read_csv(csv_file_path)
    for idx in range(len(df)):
        if idx >=100 and idx <=180:
            df.loc[idx,"height"]=df.loc[idx,"height"]+50
            df.loc[idx,"speed"]=df.loc[idx,"speed"]*0.5
            df.loc[idx,"angle"]=df.loc[idx,"angle"]-10
    ffn,_1= os.path.splitext(fn)
    # print(df)
    df.to_csv(os.path.join(test_set_path,ffn+"_test.csv"),index=False)
    # break