import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
def parse_datetime(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

def resample_flight_data(flight_data):
    print(type(flight_data))
    flight_data['UTC TIME'] = pd.to_datetime(flight_data['UTC TIME'])

    # 设置时间为索引
    original_columns = flight_data.columns.tolist()
    flight_data.set_index('UTC TIME', inplace=True)
    # print(original_columns)
    # print(flight_data)
    # 生成重采样的时间范围
    # start_time = flight_data.index.min()
    # end_time = flight_data.index.max()
    # print("航线起飞时间：",start_time)
    # print("航线降落时间：",end_time)
    # time_range = pd.date_range(start=start_time, end=end_time, freq='30S')

    # 重采样并插值
    
    # resampled_df = flight_data.reindex(time_range)
    resampled_df = flight_data.resample('30S').first()
    interpolation_columns = ['Time','height', 'speed', 'angle', 'longitude', 'latitude']
    # resampled_df[interpolation_columns] = resampled_df[interpolation_columns].interpolate(method='time')
    resampled_df[interpolation_columns] = resampled_df[interpolation_columns].interpolate(method='linear')
         
    # 对 'anum' 和 'fnum' 列填充前面的值
    resampled_df[['anum', 'fnum']] = flight_data[['anum', 'fnum']].reindex(resampled_df.index, method='nearest')

    # resampled_df[['anum', 'fnum']] = resampled_df[['anum', 'fnum']].fillna(method='ffill')

    origin_order=["Time","UTC TIME","anum","fnum","height","speed","angle","longitude","latitude"]
    # 将UTC TIME列移回原来的位置
    resampled_df.reset_index(inplace=True)
    # resampled_df=resampled_df[origin_order]
    print(resampled_df)
    resampled_df = resampled_df[original_columns]
    return resampled_df



file_cnt=0
base_path="./CA_data/"
pkl_fn="CA1883"
csv_path=os.path.join(base_path,pkl_fn+"C")
data_dict={}
csv_cnt=0
resample_path=os.path.join(base_path,pkl_fn+"R")
bnum=10001

for fn in os.listdir(csv_path):
    csv_file_path=os.path.join(csv_path,fn)
    # print(pd.read_csv(csv_file_path))
    proc_data=resample_flight_data(pd.read_csv(csv_file_path))
    proc_data.to_csv(os.path.join(resample_path,fn),index=False)
    