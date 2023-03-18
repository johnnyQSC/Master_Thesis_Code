import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#讀取資料集
fake_data=pd.read_csv('chinese_fake.csv')
true_data=pd.read_csv('chinese_true.csv')
print("讀取資料集完成")
#資料合併
merge_data=pd.concat((true_data,fake_data))
merge_data=shuffle(merge_data)
print("合併完成前後的前5筆資料: ",merge_data.head())
#刪除欄位
merge_data=merge_data.drop(["subject","time"],axis=1)
#觀察合併後資料平衡的狀況
sns.countplot(merge_data["label"])
plt.show()
#label欄位資料轉換(fake:1,true:0)
merge_data["label"]=merge_data.label.map({'Real':0,'Fake':1})
sns.countplot(merge_data['label'])
plt.show()
#另存檔案
merge_data.to_csv("TFC_Preprocess.csv",index=False,encoding="utf-8-sig")