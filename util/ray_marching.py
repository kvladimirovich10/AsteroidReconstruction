import numpy as np


def generate_grid(n, side):
    grid = np.zeros((n * n, 3))
    
    step = side / (n - 1)
    
    for i in range(n):
        for j in range(n):
            grid[j + i * n][0] = -side / 2 + j * step
            grid[j + i * n][1] = -side / 2 + i * step
            grid[j + i * n][2] = 0
    
    return grid


# plane = generate_grid(9, max(ell.a, ell.b, ell.c))
# plane_vector = np.array([0, 0, 1])

vector = np.array([5, 5, 5])


