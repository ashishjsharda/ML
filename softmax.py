import numpy as np

def softmax(x):
    """Compute softmax values for each row of x."""
    # Subtract the maximum from each element to avoid overflow
    x = x - np.max(x, axis=1, keepdims=True)
    # Exponentiate each element
    exp_x = np.exp(x)
    # Normalize by dividing by the sum of exponentiated elements
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
softmax_z = softmax(z)
print(softmax_z)
