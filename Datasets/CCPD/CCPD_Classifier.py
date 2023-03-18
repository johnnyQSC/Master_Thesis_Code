#資料分析套件載入
import pandas as pd
import numpy as np
#自然語言處理套件載入
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
#資料視覺化套件載入
import matplotlib.pyplot as plt
import seaborn as sns
#資料分割套件載入
from sklearn.model_selection import train_test_split
#讀取資料集
data=pd.read_csv('CCPD_EDA.csv')
print("讀取資料集完成")
#觀查label欄位資料平衡的狀況
sns.countplot(data["label"])
plt.show()
#檢查遺漏值數量
print("檢查遺漏值數量: ",data.isna().sum())
#label欄位資料轉換(fake:1,true:0)
mapping={'FAKE':1,'REAL':0}
data=data.replace({'label': mapping})
print(data.head())
#自然語言處理-字詞正則化
def irrelevant(text):
    return re.sub('[^a-zA-Z0-9]',' ',text)
data['text']=data['text'].apply(irrelevant)
print(data['text'].head())
print("字詞正則化-完成")
#自然語言處理-轉換所有字母為小寫
def lowering(text):
    return str(text).lower()
data['text']=data['text'].apply(lowering)
print(data['text'].head())
print("轉換所有字母為小寫-完成")
#自然語言處理-斷詞
def token(text):
    return word_tokenize(text)
data['text']=data['text'].apply(token)
print(data['text'].head())
print("斷詞-完成")
#自然語言處理-移除停用字
stopwords=set(stopwords.words('english'))
def stop_words(text):
    return [item for item in text if item not in stopwords]
data['text']=data['text'].apply(stop_words)
print(data['text'].head())
print("移除停用字-完成")
#自然語言處理-詞型還原(lemmatization)
wordnet=WordNetLemmatizer()
def lemmatization(text):
    return [wordnet.lemmatize(i,pos='v') for i in text]
data['text']=data['text'].apply(lemmatization)
print(data['text'].head())
corpus=[]
for i in data['text']:
    txt=' '.join([row for row in i])
    corpus.append(txt)
corpus[:4]
print("詞型還原完成")
#自然語言處理-詞向量轉換(TfidfVectorizer)
tfidf=TfidfVectorizer()
vector_fit=tfidf.fit_transform(corpus).toarray()
print(vector_fit.shape)
print("詞向量轉換完成")
#定義X與Y
Y=data["label"] #Fake或True
X=vector_fit #text的詞向量
print("X的維度: ",X.shape)
print("Y的維度: ",Y.shape)
#切割資料
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
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
#繪製支持向量機混淆矩陣
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
confusion_mat=metrics.confusion_matrix(Y_test,svm_prediction,labels=None,sample_weight=None)
plot_confusion_matrix(confusion_mat)