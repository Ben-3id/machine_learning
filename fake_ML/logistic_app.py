from fake import LogisticRegression 
from sklearn.linear_model import LogisticRegression as LR
import seaborn as  sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report

pd.set_option('future.no_silent_downcasting', True)

data=pd.read_csv("Raisin_Dataset.csv")
data["Class"]=data["Class"].replace({"Kecimen":1 , "Besni":0}).astype(int)



X=data.drop("Class",axis=1)
y=data["Class"]

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42,shuffle=True)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



model=LogisticRegression(lr=0.001,iter=1000,alpha=0.001,penalty_train="l2")
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))


model=LR()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))