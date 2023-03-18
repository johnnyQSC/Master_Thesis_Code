import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#讀取資料集
fake_data=pd.read_csv('ISOT_Fake.csv')
true_data=pd.read_csv('ISOT_True.csv')
print("讀取資料集完成")
#給予資料標籤(1:Fake,0:True)
true_data["label"]=np.zeros(len(true_data),dtype=int)
fake_data["label"]=np.ones(len(fake_data),dtype=int)
#資料合併
merge_data=pd.concat((true_data,fake_data))
merge_data=shuffle(merge_data)
print("合併完成前後的前5筆資料: ",merge_data.head())
#觀察合併後資料平衡的狀況
sns.countplot(merge_data["label"])
plt.show()
#另存檔案
merge_data.to_csv("ISOT_Merge.csv",index=False,encoding="utf-8")
