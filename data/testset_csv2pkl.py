import numpy as np
import os
import csv
import pickle
from sklearn.preprocessing import MinMaxScaler
import random
# from .flags_config    import HEIGHT_MAX



file_cnt=0;
base_path="./CA_data/"
ds_name="CA1883"
csv_name=ds_name+'T'
csv_path=os.path.join(base_path,csv_name)

data_dict={}
csv_cnt=0


# 在处理数据集的时候，提取出经纬度的最值，用于config_flags.py模块设定参数

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
# timestamp bnum height speed angle longitude latitude
bnum=10001

for fn in os.listdir(csv_path):
    csv_file_path=os.path.join(csv_path,fn)
    l_l_msg=[]
    ct=0
    with open (csv_file_path,"r") as f:
        
        csvReader=csv.reader(f)
        for row in csvReader:
            # row=row[2:]

            ct=ct+1
            if ct==1:
                continue
            """get the max\min value for ROI"""
            lon_max=max(lon_max,float(row[7]))
            lon_min=min(lon_min,float(row[7]))
            lat_max=max(lat_max,float(row[8]))
            lat_min=min(lat_min,float(row[8]))
            hgt_max=max(hgt_max,float(row[4]))
            hgt_min=min(hgt_min,float(row[4]))
            spd_max=max(spd_max,float(row[5]))
            spd_min=min(spd_min,float(row[5]))
            agl_max=max(agl_max,float(row[6]))
            agl_min=min(agl_min,float(row[6]))
            


for fn in os.listdir(csv_path):
    csv_file_path=os.path.join(csv_path,fn)
    l_l_msg=[]
    ct=0
    # if(bnum%10000>6):
    #     break
    print(bnum)
    with open (csv_file_path,"r") as f:
        
        csvReader=csv.reader(f)
        for row in csvReader:
            ct=ct+1
            if ct==1:
                continue
            # row=row[2:]
            # print(type(float(row[4])-hgt_min))
            l_l_msg.append([int(float(row[0])),
                            (bnum),(float(row[4])-hgt_min)/(hgt_max-hgt_min),
                            (float(row[5])-spd_min)/(spd_max-spd_min),(float(row[6])-agl_min)/(agl_max-agl_min),
                            (float(row[7])-lon_min)/(lon_max-lon_min),(float(row[8])-lat_min)/(lat_max-lat_min)])
            # print(row)
    # print(l_l_msg)
    m_msg=np.array(l_l_msg)
    # print(m_msg)
    # print("==============================")
    data_dict[csv_cnt]=m_msg
    # print(data_dict[csv_cnt])
    # print("--------------------------------")
    csv_cnt=csv_cnt+1
    bnum=bnum+1


test_pkl_fn=ds_name+"_test.pkl"



# print(train_data,valid_data,test_data)

# print(data_keys)

#save to pkl


output_path=os.path.join(base_path,test_pkl_fn)
with open(output_path,'wb') as f:
    pickle.dump(data_dict,f)


