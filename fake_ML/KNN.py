import numpy as np
import pandas as pd
from collections import Counter


class KNN:
    def __init__(self,k_neighbors,dimensions):
        self.k_neighbors=k_neighbors
        self.dimensions=dimensions
        self.X_train=None
        self.y_train=None
    
    def fit(self,X_train,y_train):
        self.X_train=np.array(X_train)
        self.y_train=np.array(y_train)

    def __Eu_Distance(self,point,X):
        return np.sum((point - X)**2)**0.5
    
    def y_hat(self,point):
        n_feat=point.shape[0]

        iter=int(n_feat/self.dimensions)
        n_3d= n_feat % self.dimensions > 0
        array_distance=[]
        j=-1
        for x in self.X_train:
            i=0
            j+=1
            values=[]
            values.append(j)
            for _ in range(iter+1*n_3d):
                values.append(self.__Eu_Distance(point[i:i+self.dimensions],x[i:i+self.dimensions]))
                i+=3
            array_distance.append(values)
        array_distance=pd.DataFrame(array_distance)
        array_distance=array_distance.rename(columns={0:"index"})
        array_distance=array_distance.sort_values(by=[i for i in range(1,iter+ 1 + 1 * n_3d)])
        indx=array_distance["index"].iloc[:self.k_neighbors]
        labels=self.y_train[indx]
        label =Counter(labels).most_common(1)[0][0]
        return label

    def predict(self,targets):
        y_pred=[]
        for target in targets:
            y_pred.append(self.y_hat(target))
        return np.array(y_pred) 