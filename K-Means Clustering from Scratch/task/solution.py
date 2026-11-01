import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):
    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


class CustomKMeans:

    def __init__(self, k):
        self.k = k
        self.centers = None

    def _find_nearest_center(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centers, axis=2)

        return np.argmin(distances, axis=1)

    def _calculate_new_centers(self, X, labels):
        return np.array([
            X[labels == k].mean(axis=0)
            for k in range(self.k)
        ])

    def fit(self, X, eps=1e-6):
        self.centers = X[:self.k].copy()

        while True:

            # Assign objects to nearest centers
            labels = self._find_nearest_center(X)

            # Compute new centers
            new_centers = self._calculate_new_centers(X, labels)

            # Check convergence - general
            shift = np.linalg.norm(new_centers - self.centers)

            if shift <= eps:
                self.centers = new_centers
                break

            self.centers = new_centers

            # # Check convergence - precise <-- suggestions
            # if np.all(
            #         np.linalg.norm(new_centers - self.centers, axis=1) <= eps
            # ):
            #     self.centers = new_centers
            # break

        return self

    def predict(self, X):
        return self._find_nearest_center(X)

    def inertia(self, X):
        labels = self.predict(X)

        return np.sum(
            (X - self.centers[labels]) ** 2
        )


if __name__ == '__main__':
    # Load data
    data_wine = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data_wine

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # # Stage1
    # centroids = X_full[:3]
    #
    # result = find_nearest_center(X_full[:], centroids).tolist()
    #
    # ## results for Stage 1
    # print(result)
    #
    # # Stage2
    #
    # # Initial centers
    # centers = X_full[:3]
    #
    # # Assign every observation to its nearest center
    # labels = find_nearest_center(X_full, centers)
    #
    # # Calculate new centers
    # new_centers = calculate_new_centers(X_full, labels).flatten().tolist()
    #
    # # Results for Stage 2
    # print(new_centers)

    # # Stage3
    #
    # model = CustomKMeans(k=2)
    #
    # model.fit(X_full)
    #
    # predictions = model.predict(X_full[:10]).tolist()
    #
    # print(predictions)

    # Stage4

    errors = []

    for k in range(2, 11):

        model = CustomKMeans(k=k)
        model.fit(X_full)

        errors.append(model.inertia(X_full))

    errors = np.array(errors).tolist()

    print(errors)

    plt.figure(figsize=(8, 5))

    plt.plot(range(2, 11), errors, marker='o')

    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)

    plt.show()