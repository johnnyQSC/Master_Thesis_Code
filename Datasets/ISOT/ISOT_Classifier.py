#資料分析套件載入
import pandas as pd
import numpy as np
#自然語言處理套件載入
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
#資料視覺化套件載入
import matplotlib.pyplot as plt
#資料分割套件載入
from sklearn.model_selection import train_test_split
#讀取資料集
merge_data=pd.read_csv('ISOT_EDA.csv')
#觀察新聞的特徵
print("觀察新聞的特徵: ",merge_data["subject"].value_counts())
#檢查遺漏值數量
print("檢查遺漏值數量: ",merge_data.isna().sum())
#資料轉換:LableEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
merge_data['subject']=labelencoder.fit_transform(merge_data["subject"])
#刪除"date"欄位
drop_data=merge_data.drop("date",axis=1)
print("刪除data欄位前5筆資料: ",drop_data.head())
#自然語言處理-詞型還原(lemmatization)
#詞型還原
print("詞型還原測試-開始")
wordnet=nltk.WordNetLemmatizer()
pattern="[^a-zA-Z]"
text=[]
for txt in drop_data.text:
    txt=re.sub(pattern," ",txt) #字詞正則化
    txt=txt.lower() #轉換為小寫
    txt=nltk.word_tokenize(txt)  #斷詞
    txt=[wordnet.lemmatize(word) for word in txt] #詞型還原
    txt=" ".join(txt) #連接字詞
    text.append(txt)
print(text[1]) 
print("詞型還原測試-成功")#內文詞型還原測試
#自然語言處理-詞向量轉換(TfidfVectorizer)
print("詞向量轉換-開始")
tfidfvector=TfidfVectorizer(stop_words='english',max_features=1000)
tfidf_text=tfidfvector.fit_transform(text)
print("詞向量轉換-完成")
#建立稀疏矩陣
print("建立稀疏矩陣-開始")
text_matrix=tfidf_text.toarray()
print("建立稀疏矩陣-完成")
#刪除title與text欄位，剩下LabelEncoding的特徵資料
train_data=drop_data.drop(["title","text"],axis=1)
print(train_data.info())
#觀察資料維度
print("訓練資料維度: ",train_data.shape)
print("內文稀疏矩陣維度: ",text_matrix.shape)
#定義X與Y
Y=train_data["label"] #Fake或True
X=text_matrix #內文稀疏矩陣
print("X的維度: ",X.shape)
print("Y的維度: ",Y.shape)
#切割資料
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
print("X訓練集維度: ",X_train.shape)
print("X測試集維度: ",X_test.shape)
print("Y訓練集維度: ",Y_train.shape)
print("Y測試集維度: ",Y_test.shape)
#決策樹實測
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
dt_model=DecisionTreeClassifier()
dt_train=dt_model.fit(X_train,Y_train)
dt_prediction=dt_train.predict(X_test)
print("-----決策樹分類報告-----")
print(classification_report(Y_test,dt_prediction,digits=4))
#隨機森林實測
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier()
rf_train=rf_model.fit(X_train,Y_train)
rf_prediction=rf_train.predict(X_test)
print("-----隨機森林分類報告-----")
print(classification_report(Y_test,rf_prediction,digits=4))
#支持向量機實測
from sklearn.svm import SVC
svm_model=SVC()
svm_train=svm_model.fit(X_train,Y_train)
svm_prediction=svm_train.predict(X_test)
print("-----支持向量機分類報告-----")
print(classification_report(Y_test,svm_prediction,digits=4))
#邏輯斯迴歸實測
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_train=log_model.fit(X_train,Y_train)
log_prediction=log_train.predict(X_test)
print("-----邏輯斯迴歸分類報告-----")
print(classification_report(Y_test,log_prediction,digits=4))
#極限梯度提升實測
from xgboost import XGBClassifier
xgb_model=XGBClassifier()
xgb_train=xgb_model.fit(X_train,Y_train)
xgb_prediction=xgb_train.predict(X_test)
print("-----極限梯度提升分類報告-----")
print(classification_report(Y_test,xgb_prediction,digits=4))
#繪製隨機森林混淆矩陣
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.GnBu)
    plt.colorbar()
    for x in range(len(confusion_mat)):
        for y in range(len(confusion_mat)):
            plt.annotate(confusion_mat[x, y], xy=(x, y),horizontalalignment='center',verticalalignment='center')
    plt.title('Confusion Matrix')    
    plt.ylabel('True label')         
    plt.xlabel('Predicted label')     
    tick_marks = np.arange(2)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.show()
from sklearn import metrics
confusion_mat=metrics.confusion_matrix(Y_test,rf_prediction,labels=None,sample_weight=None)
plot_confusion_matrix(confusion_mat)