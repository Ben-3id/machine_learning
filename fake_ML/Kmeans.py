import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

class KMeans:
    def __init__(self, K, max_iter=1000, plot_step=None):
        self.max_iter = max_iter
        self.K = K
        self.plot_step = plot_step
        self.clusters = [[] for _ in range(K)]
        self.centroids = []

    def predict(self, X):
        self.X = np.array(X)
        self.n_sample, self.n_feat = self.X.shape
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

        return self._get_label(self.clusters)

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            idx_cluster = self._closest_centroids(sample)
            clusters[idx_cluster].append(idx)
        return clusters
    
    def get_WCSS(self, clusters, centroids):
        total = 0.0
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                distance = euclidean_distance(self.X[sample_idx], centroids[idx])
                total += distance ** 2
        return total

    def _closest_centroids(self, point):
        dis_cluster = [euclidean_distance(point, centroid) for centroid in self.centroids]
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
    
    def _get_label(self, clusters):
        labels = np.empty(self.n_sample)
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = idx
        return labels
    
    def choice(self):
        indices = np.random.choice(self.n_sample, self.K, replace=False)
        return self.X[indices]

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cluster in enumerate(self.clusters):
            points = self.X[cluster]
            ax.scatter(points[:, 0], points[:, 1])
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()

    # NEW: elbow method
    @staticmethod
    def elbow(X, k_range=range(1, 11), plot=True):
        WCSS = []
        for k in k_range:
            model = KMeans(K=k)
            model.predict(X)
            WCSS.append(model._WCSS)

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(k_range, WCSS, marker="o")
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
            plt.title("Elbow Method for Optimal K")
            plt.xticks(k_range)
            plt.grid(True)
            plt.show()

        # Simple elbow guess: look for max drop ratio
        deltas = np.diff(WCSS)
        if len(deltas) > 0:
            best_k = k_range[np.argmin(deltas) + 1]  # index + 1 since diff reduces length
        else:
            best_k = 1

        return best_k, WCSS
