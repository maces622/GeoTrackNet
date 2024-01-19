import matplotlib.pyplot as plt
import os
import sys
import csv
import numpy as np

file_cnt=0
base_path="./CA_data/"
pkl_fn="CA1803R"
csv_path=os.path.join(base_path,pkl_fn)
data_dict={}
csv_cnt=0
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
            # print(type(float(row[4])-hgt_min))
            l_l_msg.append([int(float(row[1])),
                            (bnum),(float(row[4])),
                            (float(row[5])),(float(row[6])),
                            (float(row[7])),(float(row[8]))])
        
        if(ct<1000):
            print(csv_file_path)
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
    
for i in range(len(data_dict)):
    now=data_dict[i]
    # print(len(now))
    ext_ts= range(0,len(now))
    ext_hg=now[:, 6]
    plt.plot(ext_ts,ext_hg)
    
# plt.legend()
plt.title('Flight Altitude Over Time')
plt.xlabel('Time')
plt.ylabel('Altitude')
plt.savefig("hgt.png")
