import pandas as pd
import pickle as pkl
 
with open ('./mean.pkl','rb') as file:
    data=pkl.load(file)
print(len(data[1]))

# df=pd.DataFrame(data[5])
# print(df)
# df.to_csv('output_file2.csv',index=False,header=False)
