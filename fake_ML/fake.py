import numpy as np
from collections import Counter


class LinearRegression:

    def __init__(self,lr=0.001,iter=1000,alpha=0):
        self.lr=lr
        self.iter=iter
        self.w =None
        self.bias=None
        self.alpha=alpha
            

    def fit(self, X_train ,y_train):
        n_samp , n_feat=X_train.shape
        self.w=np.zeros(n_feat)
        self.bias=0
        
        for i in range(self.iter):
           y_train_pred=np.dot(X_train,self.w)+self.bias

           dw=(1/n_samp)*np.dot(X_train.T,(y_train_pred-y_train)) + self.alpha * np.sum(abs(self.w))
           db=(1/n_samp)*np.sum((y_train_pred-y_train))+ self.alpha *np.sum(abs(self.w))
           self.w -= self.lr * dw
           self.bias -= self.lr * db


    def pred(self, X_train):
        y_train_pred=np.dot(X_train,self.w)+self.bias
        return y_train_pred
    
def sigmoid(z):
    # prevents overflow
      return   1 / (1 + np.eX_trainp(-1*z))


class LogisticRegression:
    def __init__(self,lr=0.001,iter=1000 , alpha=0,penalty_train=None):
        self.lr=lr
        self.iter=iter
        self.w =None
        self.bias=None
        self.alpha=alpha
        self.penalty_train=penalty_train
    


    def feed_prob(self,X_train):
        z=np.dot(X_train,self.w) + self.bias
        return sigmoid(z)
    

    def fit(self,X_train,y_train):
        n_sample , n_feat = X_train.shape
        self.w=np.zeros(n_feat)
        self.bias=0
        for i in range(self.iter):
            y_trainhat=self.feed_prob(X_train)
            dw=(1 / n_sample) * np.dot(X_train.T , (y_trainhat - y_train))

            if self.alpha > 0:
                if self.penalty_train == "l1":
                    dw += self.alpha * np.sign(self.w)
                elif self.penalty_train == "l2":
                    dw += self.alpha * self.w

            db=(1 / n_sample) * np.sum(y_trainhat - y_train) 
            
            self.w -= self.lr * dw
            self.bias -= self.lr * db

    def score(self,X_train,y_train):
        y_trainhat=self.predict(X_train)
        return np.mean(y_trainhat==y_train)

    def predict(self,X_train):
        y_train_hat=self.feed_prob(X_train)
        y_train_pred=[0 if y_train <= 0.5 else 1 for y_train in y_train_hat]
        return y_train_pred
    def loss(self,X_train,y_train):
        y_train_hat=self.predict(X_train)
        loss = - y_train * np.log(y_train_hat) - (1-y_train) * np.log(1-y_train_hat)
        return loss
    





class KNN:
    def __init__(self,n_neighbors=3):
        self.n_neighbors=n_neighbors

    def fit(self, X_train  , y_train ):
        self.X_train=np.array(X_train)
        self.y_train=np.array(y_train)
    
    def _get_distance(self , a, b ):
        return np.sum((a - b) ** 2 ) ** 0.5
    
    def predict(self,points):
        points=np.array(points)
        n_sample=points.shape[0]
        target=[]
        for i in range(n_sample):
            distance=[self._get_distance(points[i],X) for X in self.X_train]
        
            neighbors=np.argsort(distance)[:self.n_neighbors]
            labels=[self.y_train[j] for j in neighbors]

            target.append(Counter(labels).most_common(1)[0][0])
        return np.array(target)

    