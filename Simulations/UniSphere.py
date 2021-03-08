import numpy as np 
def UniSphere(n, d):
    pts = np.random.randn(n, d)
    for i in range(n):
        pts[i] = pts[i]/np.linalg.norm(pts[i])
    return pts