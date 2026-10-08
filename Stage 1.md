# Stage 1: Where do you belong

## Description

Let's discuss a few things before we start. We use the [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset) provided by the `sklearn` package. Initially, it  
contained the `target` variable that is not used by us in the clustering task and will be the result for clustering  
(since there are groups to predict, it is natural to suppose that some "optimal" clustering would be these groups).  
So, we have removed it for you already.

We also need you to provide the desired outputs as Python lists so that our tests would work correctly whenever there  
is more than one value to be printed. You can always make `list(...)` of your `np.ndarray` to convert it to a list.

The first step of the K-Means loop (excluding the very first step) is to determine to which cluster every object  
belongs; in other words, to calculate the distances to each cluster's center and find the minimal one. We mean the  
**Euclidean distance** whenever we say "distance."

We would, later on, start wrapping everything into a single class (we are going to name it `CustomKMeans`), but you can  
begin creating one right now to make this function a class method. Alternatively, you can make it a standalone function;  
we would suggest naming it `find_nearest_center`. Choose whatever you find appropriate.

We expect the function to have the features array and the centroids' positions as input and return an array of the  
nearest center's labels for each input object.

## Objectives
- Create the `find_nearest_center` function.
- Print the result of your function for the last ten objects of the `X_full` array. Take the first three objects of  
  the `X_full` array as the clusters' centers.

## Examples
### Example 1: an example of the output
```
[0, 1, 2, 2, 1, 0, 0, 1, 2, 2]
```

## Results
### 1
This version because it mirrors the K-Means algorithm:
for each point → compute distances to centers → choose nearest center.
```
def find_nearest_center(X, centers):
    labels = []

    for point in X:
        distances = np.linalg.norm(centers - point, axis=1)
        labels.append(np.argmin(distances))

    return np.array(labels)
```

### 2
A very NumPy-friendly implementation (with broadcasting) is:
```
def find_nearest_center(X, centers):
    distances = np.linalg.norm(
        X[:, np.newaxis] - centers,
        axis=2
    )

    return np.argmin(distances, axis=1)
```