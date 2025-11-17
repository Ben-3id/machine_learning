import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from fake import Gaussian as G
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as CM
data=pd.read_csv('D:\\coding\\fake_ML\\Files\\data_10.csv').drop(['010-000-024-033','010-000-030-096','020-000-032-221','020-000-033-111'],axis=1)
'''training data set'''
X_train=(data[:4500].drop(['anomaly'],axis=1))
y_train=(data.loc[:4500,['anomaly']])
'''testing data set'''
X_test=(data[4500:].drop(['anomaly'],axis=1))
y_test=(data.loc[4500:,['anomaly']])

scaler = StandardScaler()
X_train = (scaler.fit_transform(X_train))
X_test= scaler.transform(X_test)

mu = np.mean(X_train, axis=0)
cov = np.cov(X_train, rowvar=False)
gaussian_model = multivariate_normal(mean=mu, cov=cov,allow_singular=True)

model=G()
model.fit(X_train,y_train)
print(classification_report(y_test,model.predict(X_test)))
