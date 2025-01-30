import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

class Clustering:
    def __init__(self, data, k=-1):
        # Define data
        self.data = data
        # Run the KMeans clustering algorithm
        best_k, labels, kmeans, metrics = self.find_optimal_k(k)
        # Define the output variables
        self.best_k = best_k
        self.labels = labels
        self.kmeans = kmeans
        self.metrics = metrics

    # Support functions for clustering analysis
    def plot_data_points(self, labels, kmeans, ax=None):
        """
        Plot the data points with colored clusters.

        Parameters:
        - data: np.ndarray, shape (n_samples, n_features), raw data points
        - labels: np.ndarray, shape (n_samples,), cluster labels for the data points
        - kmeans: KMeans instance, fitted KMeans model
        - ax: matplotlib.axes.Axes, optional, axis to plot on

        Returns:
        - ax: matplotlib.axes.Axes, axis with the plot
        """
        # Define data
        data = self.data
        # Check the dimensions of the data
        if data.shape[1] != 2:
            return
        # If no axis is provided, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot the data points with colored clusters
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
        # Plot the cluster centers
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x')
        
        # Set axis labels
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Display the plot
        plt.show()

    # Silhouette and Elbow Method plots
    def plot_metrics(self, metrics):
        """
        Plot the distortion and silhouette scores for different k values.

        Parameters:
        - metrics: dict, dictionary with distortions and silhouette scores for each k
        """
        # Create a new figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the distortion
        ax[0].plot(metrics["k_values"], metrics["distortions"], marker='o')
        ax[0].set_xlabel("Number of Clusters (k)")
        ax[0].set_ylabel("Distortion")
        ax[0].set_title("Elbow Method")

        # Plot the silhouette scores
        ax[1].plot(metrics["k_values"], metrics["silhouette_scores"], marker='o')
        ax[1].set_xlabel("Number of Clusters (k)")
        ax[1].set_ylabel("Silhouette Score")
        ax[1].set_title("Silhouette Score")

        # Display the plot
        plt.show()

    # The function takes data and number of clusters, then returns the kmeans instance, cluster labels, metrics
    def kmeans_metrics(self, k):
        # Define data
        data = self.data
        # Reshape the data to be able to use KMeans
        data = data.T if data.shape[0] < data.shape[1] else data
        # Fit the KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        # Find the distortion and silhouette score
        distortion = kmeans.inertia_
        silhouette = silhouette_score(data, labels)
        # Return the kmeans instance, cluster labels, distortion, and silhouette score
        return kmeans, labels, distortion, silhouette

    # The function finds the local maximas of the data
    def find_local_maxima(self, dataX, dataY):
        """
        Find the indices of local maxima in a NumPy array.

        Parameters:
        data (numpy.ndarray): The input data array.

        Returns:
        numpy.ndarray: The indices of the local maxima.
        """
        # Ensure the input is a NumPy array
        dataX = np.asarray(dataX)
        dataY = np.asarray(dataY)

        # Identify local maxima
        # We can check if the value is greater than the previous and next values
        local_maxima = (dataY[1:-1] > dataY[:-2]) & (dataY[1:-1] > dataY[2:])
        
        # Add 1 to the indices to account for the offset caused by slicing
        local_maxima_indices = np.where(local_maxima)[0] + 1
        
        # Sort the indices by corresponding y values in descending order
        local_maxima_indices = local_maxima_indices[np.argsort(dataY[local_maxima_indices])[::-1]]
        # Find the corresponding x values
        local_maxima_x = dataX[local_maxima_indices]
        # Find the corresponding y values
        local_maxima_y = dataY[local_maxima_indices]

        if len(local_maxima_indices) == 0:
            return 0, dataX[0], dataY[0]
        else:
            return local_maxima_indices, local_maxima_x, local_maxima_y

    # Function to determine optimal k using the Elbow Method and Silhouette Score
    # The detailed information for Silhouette Score can be found in the following link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    # The function returns the optimal k, cluster labels, KMeans instance, and metrics
    def find_optimal_k(self, max_k=-1):
        """
        Determine the optimal number of clusters using the Elbow Method and Silhouette Score.

        Parameters:
        - data: np.ndarray, shape (2, n_samples), raw data points
        - max_k: int, maximum number of clusters to test

        Returns:
        - k_optimal: Optimal number of clusters
        - labels: Cluster labels for the data points
        - kmeans: Fitted KMeans instance
        - metrics: Dictionary with distortions and silhouette scores for each k
        """
        # Define data
        data = self.data
        # First, we need to determine the maximum number of clusters to test
        if max_k == -1:
            max_k = 10

        # Define arrays to store metrics
        distortions = np.zeros(max_k - 1)
        silhouette_scores = np.zeros(max_k - 1)
        k_values = np.array(range(2, min(max_k, len(data)) + 1))
        
        # Initialize variables to store the best k and corresponding metrics
        best_k = None
        best_labels = None
        best_kmeans = None

        # For each k value, fit the KMeans model and compute metrics
        for i, k in zip(range(len(k_values)), k_values):
            try:
                # Perform KMeans clustering
                kmeans, labels, distortion, silhouette = self.kmeans_metrics(k)
                # Store the metrics
                distortions[i] = distortion
                silhouette = silhouette_score(data, labels)
                silhouette_scores[i] = silhouette
            
            except ValueError as e:
                continue
        
        # The metrics for silhouette score and distortion are returned for analysis
        # However, the best k might be found at the beginning of the range which is not desired
        # Thus, we need to check if any near best k exists with a lower distortion
        # Find peak silhouette scores
        peak_indices, peak_kValues, peak_silhouette_scores = self.find_local_maxima(k_values, silhouette_scores)
        if type(peak_indices) == int:
            best_k = peak_kValues
        else:
            best_k = peak_kValues[0]

        best_kmeans = KMeans(n_clusters=best_k, random_state=42).fit(data)
        best_labels = best_kmeans.labels_
        
        # Ensure we have a valid k
        if best_k is None:
            raise ValueError("Could not determine an optimal k. Check the data or clustering setup.")
        
        # Return metrics for analysis and plotting
        metrics = {
            "k_values": list(k_values),
            "distortions": distortions,
            "silhouette_scores": silhouette_scores,
        }

        return best_k, best_labels, best_kmeans, metrics

    def clustering_results(self):
        return self.best_k, self.labels, self.kmeans, self.metrics

# Example usage
# Define data
d, n = 20, 2
data = np.random.rand(d, n)

# Find the optimal number of clusters
clustering = Clustering(data)
k_optimal, labels, kmeans, metrics = clustering.clustering_results()

print("Optimal number of clusters:", k_optimal)
clustering.plot_data_points(labels, kmeans)
clustering.plot_metrics(metrics)