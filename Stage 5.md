# Stage 5/6: Automation is the Key
## Description
Remember we told about the silhouette score in the previous step? It is now time to use it, too, to double-check the k  
you determined in the previous step.

We need you to use the `sklearn.metrics.silhouette_score` to calculate the score for different `k`. Since the higher the  
silhouette score is, the better the clustering is (for our case at least), we need to determine the maxima.

Do the same range for `k` you did in the previous step. In case the `k` you get is the same as at the previous step, you  
should feel safe about your choice.

## Objectives

1. Use the custom K-Means class with k ranging from **2** to **10** (both included) and calculate the silhouette score  
   for each `k`.
2. Print the results for the silhouette score for each of the `k` values.
3. To find the desired `k`, plot the dependence of the silhouette score from `k` and determine the maxima. Check if it  
   is the same `k` as you had in the previous step.

## Example
### Example 1: an example of the output

```
[9, 8, 7, 6, 5, 4, 3, 2, 1]
```