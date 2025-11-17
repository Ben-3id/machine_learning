import numpy as  np
import pandas as pd
import seaborn as  sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from fake import KMeans as my
import pandas as pd

data = pd.read_csv("clustering/basic2.csv")
X = data.drop("color", axis=1)
y=data["color"]
plt.scatter(X.iloc[:,0],X.iloc[:,1])
plt.show()

model=my(K=10,max_iter=1000,plot_step=True)
y_prd=model.predict(X)
# print(confusion_matrix(y_prd,y))
