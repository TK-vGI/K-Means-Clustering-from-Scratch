### Stage 2/6: Calculating the means
## Description
The next step of the K-Means algorithm is to calculate new coordinates for each cluster's center once you know which of  
the "old" clusters every object belongs to. To do it, you need to calculate each feature's mean value for each cluster's  
objects — and name it the "new" cluster center.

Name the new function `calculate_new_centers`. This new function, together with the previous one, would be later wrapped  
into a class. You could either implement it into the class now or make it a standalone function. You get to decide  
whatever feels better to you.

## Objectives
1. Create the `calculate_new_centers` function and calculate new centers as the mean features for each of the classes;
2. Use the labels you got in the previous stage (using the `find_nearest_center` function) to calculate new centers;
3. Print the resulting coordinates of the centers for the entire dataset. Use the `.flatten()` method to turn a 2D array  
   into 1D. Don't round values.

## Examples
### Example 1: an example of the output
```
[0.31, 0.79, 0.35, 0.45, 0.91, 0.23, 0.65, 0.31, 0.03, 0.02, 0.02, 0.57, 0.47, 0.33, 0.73, 0.81, 0.8, 0.89, 0.29 0.5,  
0.04, 0.91, 0.84, 0.08, 0.8, 0.77, 0.82, 0.82, 0.6, 0.4, 0.83, 0.97, 0.1, 0.18, 0.87, 0.73, 0.29, 0.39, 0.89]
```

## Results
### 1