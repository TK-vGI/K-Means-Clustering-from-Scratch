# Stage 6/6: The final stage
## Description

In this stage, let's challenge your code. Use optimal `k` you found in the previous steps and your custom K-Means class  
to create the predictions on clusters for each object in `X_full`. Then, you can use our function to plot your predictions  
(one 2D subdimension would do) or make yours. We asked you to guess how many clusters would be required for a great fit.  
It is now a perfect time to see whether you were correct.

The function we provided actually plots two images: one is your predictions, and the second one is the benchmark data,  
so you can compare those and estimate how well your model performs.

## Objectives

1. Using the appropriate k you got in the previous step (or by reusing your "find the appropriate `k`" function with  
   `p = 0.2` — see the previous stage), create predictions for each object in `X `using your custom K-Means class.
2. Print the results for the first 20 objects for us to test.

## Example
### Example 1: an example of the output

```
[0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0]
```