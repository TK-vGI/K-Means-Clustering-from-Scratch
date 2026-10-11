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


def find_nearest_center(features, centers):
    labels = []

    for point in features:
        distances = np.linalg.norm(centers - point, axis=1)
        labels.append(np.argmin(distances))

    return np.array(labels)


def calculate_new_centers(features, labels):
    centers = []

    for cluster_id in np.unique(labels):
        center = features[labels == cluster_id].mean(axis=0)
        centers.append(center)

    return np.array(centers)


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


    # Stage2
    # Initial centers
    centers = X_full[:3]

    # Assign every observation to its nearest center
    labels = find_nearest_center(X_full, centers)

    # Calculate new centers
    new_centers = calculate_new_centers(X_full, labels).flatten().tolist()

    # Results for Stage 2
    print(new_centers)