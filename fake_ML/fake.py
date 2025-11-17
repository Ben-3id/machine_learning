import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

class LinearRegression:

    def __init__(self,lr=0.001,iter=1000,alpha=0):
        self.lr=lr
        self.iter=int(iter)
        self.w =None
        self.bias=None
        self.alpha=alpha


    def fit(self, X_train ,y_train):
        n_samp , n_feat=X_train.shape
        self.w=np.zeros(n_feat)
        self.bias=0
        '''gradient'''
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
      return 1 / (1 + np.exp(-1 * z))


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
    
    def predict(self,points):
        points=np.array(points)
        n_sample=points.shape[0]
        target=[]
        for i in range(n_sample):
            distance=[euclidean_distance(points[i],X) for X in self.X_train]
        
            neighbors=np.argsort(distance)[:self.n_neighbors]
            labels=[self.y_train[j] for j in neighbors]

            target.append(Counter(labels).most_common(1)[0][0])
        return np.array(target)

class KMeans:
    def __init__(self,K,max_iter=1000,plot_step=None):
        self.max_iter=max_iter
        self.K=K
        self.plot_step=plot_step
        self.clusters=[[] for _ in range(K)]
        self.centroids=[]

    def predict(self, X):
        self.X = np.array(X)
        self.n_sample, self.n_feat = self.X.shape
        #idxs=np.random.choice(self.n_sample,self.K,replace=False)
        self.centroids = self.choice()
        self._WCSS = 0

        for _ in range(self.max_iter):
            self.clusters = self.create_clusters(self.centroids)

            if self.plot_step:
                self.plot()

            old_centroids = self.centroids.copy()
            self.centroids = self._get_centroids(self.clusters)

            if self._is_closest(old_centroids, self.centroids):
                break

            if self.plot_step:
                self.plot()

            self._WCSS = self.get_WCSS(self.clusters, self.centroids)

    def create_clusters(self,centroids):
        clusters=[[] for _ in range(self.K)]
        for idx,sample in enumerate(self.X):
            idx_cluster=self._closest_centroids(sample)
            clusters[idx_cluster].append(idx)
        return clusters
    
    def get_WCSS(self, clusters, centroids):
        total = 0.0
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                distance = euclidean_distance(self.X[sample_idx], centroids[idx])
                total += distance ** 2
        return total

    def _closest_centroids(self,point):
        dis_cluster=[euclidean_distance(point,centroid) for centroid in self.centroids]
        return np.argmin(dis_cluster)
    
    def _get_centroids(self, clusters):
        new_centroids = np.zeros((self.K, self.n_feat))
        for idx, cluster in enumerate(clusters):
            if cluster:  # avoid empty cluster crash
                new_centroids[idx] = np.mean(self.X[cluster], axis=0)
            else:
                new_centroids[idx] = self.X[np.random.choice(self.n_sample)]
        return new_centroids

    def _is_closest(self, old_centroids, new_centroids):
        return np.allclose(old_centroids, new_centroids)
    
    def _get_label(self,clusters):
        labels=np.empty(self.n_sample)
        for idx,cluster in enumerate(clusters):
            for sample_cluster in cluster:
                labels[sample_cluster]=idx
        return labels
    
    def choice(self):
        data = np.arange(self.n_sample)  # indices of samples
        unite = int(self.n_sample / (self.K))
        centroids = []
        for i in range(self.K):
            centroids.append(i*unite)
        return np.array(centroids).reshape(-1,1)

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

class Gaussian:
    def __init__(self):
        self.threshold=None
    
    def fit(self,X,y):
        self.mu,self.cov=self.estimate_guassian(X)
        p_val=self.compute_pval(X)
        self.threshold=self.find_threshold(p_val)

    def predict(self,X):
        p=self.compute_pval(X)
        return (p<self.threshold).astype(int)

    def estimate_guassian(self,X):
        mu=np.mean(X,axis=0)
        # ✅ CORRECT: full covariance matrix
        cov = np.cov(X, rowvar=False)

        return mu,cov
    
    def compute_pval(self,X): 
        from scipy.stats import multivariate_normal
        dist = multivariate_normal(mean=self.mu, cov=self.cov,allow_singular=True)  # Diagonal covariance for independence
        return dist.pdf(X)
    
    def find_threshold(self, p_val): # Ensure 1D

        best_epsilon = np.percentile(p_val, 1)

        return best_epsilon