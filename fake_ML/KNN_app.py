import numpy as  np
import pandas as pd
import seaborn as  sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from fake import KNN
from KNN import KNN as myKnn


data=pd.read_csv("Raisin_Dataset.csv")
pd.set_option('future.no_silent_downcasting', True)
data=data.replace({"Kecimen":1,"Besni":0})
X=data.drop("Class",axis=1)
y=data["Class"]
X_Train , X_test , y_train ,y_test =train_test_split(X,y,test_size=0.2 , random_state=12345 , shuffle=True)



le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


scaler=StandardScaler()

X_Train=scaler.fit_transform(X_Train)
X_test=scaler.transform(X_test)
y_test=y_test.astype(int)
k=50
my=myKnn(k,3)
Knn=KNN(k)
real_KNN = KNeighborsClassifier(k)


KNNS={"New_KNN":my,"fake_KNN":Knn ,"scikit-learn":real_KNN}
for name,cls in KNNS.items():
    print(f"-------------------{name} algorithm--------------------")

    cls.fit(X_Train,y_train)
    y_pred=cls.predict(X_test)
    y_pred=y_pred.astype(int)

    mat=confusion_matrix(y_test,y_pred)
    print(mat)

    score=precision_score(y_test,y_pred)
    print(f"precision_score : {score}")
    score=recall_score(y_test,y_pred)
    print(f"recall_score : {score}")
    score=f1_score(y_test,y_pred)
    print(f"f1_score : {score}")
    print("\n")


