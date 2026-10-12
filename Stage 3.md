# Stage 3/6: Get it wrapped
## Description

Create a class wrap for the K-Means algorithm in this stage. We suggest the following construction for it:

```
class CustomKMeans:
	def __init__(self, k):
		self.k = k
		self.centers = None

	def fit(self, X, eps=1e-6):
		...

	def predict(self, X):
		...
```

However, feel free to change it.

The fit method must be based on the functions you've created. Use the first `k` objects of `X_full` as the starting  
points for the cluster centers. It might not be the best solution, but let us keep it this simple.

To check whether the algorithm is finished or not at each step, use a small `eps` value (say, `eps=1e-6`) and check if  
the square distance from the "old" cluster center to a "new" one is no more than eps. You can use `numpy.linalg.norm`  
to calculate that.

The predict method is expected to take `X_full` array of object features and return a cluster label for each object.

Check out additional materials on [K-Means Clustering](https://www.youtube.com/watch?v=GZj6ikx8PAc) to recall the concept.

## Objectives
1. Implement the `fit` method for the class based on the functions from stages 1 and 2 using the first `k` objects of  
   `X_full` as the initial clusters' centers. Do not forget to check whether the algorithm is finished at each step  
   using `eps`.
2. Implement the `predict` method for the class.
3. Fit your model with `k=2` and print the predictions for the first ten objects of `X_full`.

## Examples
### Example 1: an example of the output

```
[0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
```

# Results

```
initialize centers
↓
assign labels
↓
compute new centers
↓
check convergence
↓
repeat
```

### A slightly stricter convergence check

```
if np.all(np.linalg.norm(new_centers - self.centers, axis=1) <= eps):
    break
```
instead of:
```
shift = np.linalg.norm(new_centers - self.centers)

if shift <= eps:
    break
```
the first one matches the wording of the assignment more closely because it verifies each center individually.