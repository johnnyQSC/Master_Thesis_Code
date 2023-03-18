#資料分析套件載入
import pandas as pd
import numpy as np
#自然語言處理套件載入
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#資料視覺化套件載入
import matplotlib.pyplot as plt
import seaborn as sns
#資料分割套件載入
from sklearn.model_selection import train_test_split
#讀取資料集
data=pd.read_csv('TFC_Preprocess.csv')
print("讀取資料集完成")
#觀查label欄位資料平衡的狀況
sns.countplot(data['label'])
plt.show()
#詞向量轉換(CountVectorizer,TfidfTransformer)
vectoerizer=CountVectorizer(min_df=1,max_df=1.0,token_pattern='\\b\\w+\\b')
vectoerizer=vectoerizer.fit(data['title'])
bag_of_words=vectoerizer.get_feature_names()
countvector=vectoerizer.transform(data['title']).toarray()
tfidf=TfidfTransformer()
vector_fit=tfidf.fit(countvector)
for idx, word in enumerate(vectoerizer.get_feature_names()):
  print("{}\t{}".format(word,tfidf.idf_[idx]))
tfidf=vector_fit.transform(countvector)
print("詞向量轉換完成")
#定義X與Y
Y=data["label"] #Fake或True
X=tfidf #title的詞向量
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
#繪製極限梯度提升混淆矩陣
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
confusion_mat=metrics.confusion_matrix(Y_test,xgb_prediction,labels=None,sample_weight=None)
plot_confusion_matrix(confusion_mat)