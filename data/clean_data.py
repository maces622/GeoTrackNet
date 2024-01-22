import matplotlib.pyplot as plt
import os
import sys
import csv
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
file_cnt=0
base_path="./CA_data/"
pkl_fn="CA1883"
csv_path=os.path.join(base_path,pkl_fn)
clean_path=os.path.join(base_path,pkl_fn+"C")

clean_loc="height"
pm=6;

for fn in os.listdir(csv_path):
    csv_file_path=os.path.join(csv_path,fn)
    l_l_msg=[]
    ct=0
    df=pd.read_csv(csv_file_path)
    for pin in range(1,len(df)):
        if pin >50 and pin <len(df)-50:
            avg=0.0
            sumf=0.0
            sumb=0.0
            for x in range(1,pm):
                sumf=sumf+df.loc[pin-x][clean_loc]
                sumb=sumb+df.loc[pin+x][clean_loc]
            avg=(sumf+sumb)/((pm-1)*2)
            # print(sum_bf)
            if abs(df.loc[pin][clean_loc]-avg)>avg*0.4 and abs(sumf-sumb)<avg*0.01:
                print("---------1")
                df.loc[pin,clean_loc]=avg

            elif abs(df.loc[pin][clean_loc]-avg)>avg*0.1:
                print("---------2")
                df.loc[pin,clean_loc]=avg

            elif abs(df.loc[pin][clean_loc]-df.loc[pin-1,clean_loc])>abs(sumf-sumb)*4:
                print("---------3")
                df.loc[pin,clean_loc]=df.loc[pin-1,clean_loc]
    
    df.to_csv(os.path.join(clean_path,fn),index=False)


                
        
    ext_ts= range(0,len(df))
    ext_hg=df[:][clean_loc]
    plt.plot(ext_ts,ext_hg)
    
# plt.legend()
plt.title('Flight angle Over Time')
plt.xlabel('Time')
plt.ylabel('angle')
plt.savefig("hgt.png")


# clean data 

# for i in range(len(data_dict)):
#     now_adsb=data_dict[i]
#     for pin in range(len(now_adsb)):
#         if pin >50 and pin <len(now_adsb)-50:
#             avg=0.0
#             # print(len(now_adsb))
#             sumf=0.0
#             sumb=0.0
#             for x in range(1,pm):
#                 sumf=sumf+now_adsb[pin-x,clean_loc]
#                 sumb=sumb+now_adsb[pin+x,clean_loc]
#             avg=(sumf+sumb)/((pm-1)*2)
#             # print(sum_bf)
#             if abs(now_adsb[pin,clean_loc]-avg)>avg*0.4 and abs(sumf-sumb)<avg*0.01:
#                 now_adsb[pin,clean_loc]=avg

#             elif abs(now_adsb[pin,clean_loc]-avg)>avg*0.1:
#                 now_adsb[pin,clean_loc]=avg

#             elif abs(now_adsb[pin,clean_loc]-now_adsb[pin-1,clean_loc])>abs(sumf-sumb)*2:
#                 now_adsb[pin,clean_loc]=avg