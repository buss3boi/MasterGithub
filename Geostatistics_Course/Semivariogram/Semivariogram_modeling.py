# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:28:09 2023

@author: busse
"""
import numpy as np

def semivar_mod(h, C0, C1, hR, pr, eg):
    h = np.atleast_2d(h)  # Ensure h is at least a 2D array
    gh = C0 + C1 * (1 - np.exp(-pr * (h / hR) ** eg))
    ch = C0 + C1 - gh

    row, col = ch.shape

    # Add nugget if h == 0 to include the effect of measurement uncertainty
    if row == col:
        ch += np.eye(row) * C0
    else:
        vkol, mrad = h.shape

        # Add nugget for covariance matrix (symmetric case)
        if vkol > 1:
            p, q = np.where(h == 0)
            ok = p.shape[0]
            for i in range(ok):
                ch[p[i], q[i]] += C1 + C0

        if vkol == 1:
            p, q = np.where(h == 0)
            ok = q.shape[0]
            for i in range(ok):
                ch[0, q[i]] += C1 + C0

    return ch.squeeze()  # Remove singleton dimensions for vector input

# Example usage with both vector and matrix inputs for h:
h_vector = np.array([0, 1, 2, 3])
h_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

C0 = 1
C1 = 2
hR = 0.5
pr = 0.1
eg = 0.2

result_vector = semivar_mod(h_vector, C0, C1, hR, pr, eg)
result_matrix = semivar_mod(h_matrix, C0, C1, hR, pr, eg)

print("Result for vector input:")
print(result_vector)

print("\nResult for matrix input:")
print(result_matrix)
