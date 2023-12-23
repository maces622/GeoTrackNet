import numpy as np
import os
import csv
import pickle



file_cnt=0;
base_path="./CA_data/"
pkl_fn="CA1803"
csv_path=os.path.join(base_path,pkl_fn)

data_dict={}
csv_cnt=0


# 在处理数据集的时候，提取出经纬度的最值，用于config_flags.py模块设定参数

lon_max=-90.0
lon_min=90.0
lat_max=-180
lat_min=180

# timestamp bnum height speed angle longitude latitude
bnum=10001
for fn in os.listdir(csv_path):
    csv_file_path=os.path.join(csv_path,fn)
    l_l_msg=[]
    ct=0
    with open (csv_file_path,"r") as f:
        
        csvReader=csv.reader(f)
        for row in csvReader:
            ct=ct+1
            if ct==1:
                continue
            lon_max=max(lon_max,float(row[7]))
            lon_min=min(lon_min,float(row[7]))
            lat_max=max(lat_max,float(row[8]))
            lat_min=min(lat_min,float(row[8]))
            l_l_msg.append([int(row[0]),
                            (bnum),float(row[4]),
                            float(row[5]),float(row[6]),
                            float(row[7]),float(row[8])])
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

test_pkl_fn=pkl_fn+"_test.pkl"
train_pkl_fn=pkl_fn+"_train.pkl"
valid_pkl_fn=pkl_fn+"_valid.pkl"

print(type(data_dict))

data_keys=list(data_dict.keys())
total_size=len(data_keys)
total_size = len(data_keys)
train_size = int(0.7 * total_size)
valid_size = int(0.15 * total_size)

# 打乱数据键的顺序
np.random.shuffle(data_keys)

# 分割数据键
train_keys = data_keys[:train_size]
valid_keys = data_keys[train_size:train_size + valid_size]
test_keys = data_keys[train_size + valid_size:]

train_data = {k: data_dict[k] for k in train_keys}
valid_data = {k: data_dict[k] for k in valid_keys}
test_data = {k: data_dict[k] for k in test_keys}

# print(train_data,valid_data,test_data)

# print(data_keys)

#save to pkl
output_path=os.path.join(base_path,train_pkl_fn)
with open(output_path,'wb') as f:
    pickle.dump(train_data,f)

output_path=os.path.join(base_path,test_pkl_fn)
with open(output_path,'wb') as f:
    pickle.dump(test_data,f)

output_path=os.path.join(base_path,valid_pkl_fn)
with open(output_path,'wb') as f:
    pickle.dump(valid_data,f)

print(lon_max,lon_min,lat_max,lat_min)
