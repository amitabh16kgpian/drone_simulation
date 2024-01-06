import numpy as np

def fun01(a1c, a2c, a1d, a2d, x1tar, y1tar, x2tar, y2tar):
    k1 = 0.2
    V = 20
    k2 = 1000
    u1n = 10
    u2n = 10
    
    dist1 = np.zeros(u1n)
    energy1 = np.zeros(u1n)
    disttar1 = np.zeros(u1n)
    ss1 = np.zeros(u1n)
    
    dist2 = np.zeros(u2n)
    energy2 = np.zeros(u2n)
    disttar2 = np.zeros(u2n)
    ss2 = np.zeros(u2n)
    
    for i in range(u1n):
        dist1[i] = np.linalg.norm([a1c[i] - a1d[i], a1c[i + u1n] - a1d[i + u1n]], 2)
        energy1[i] = k1 * V**2 * dist1[i]
        disttar1[i] = np.linalg.norm([a1d[i] - x2tar, a1d[i + u1n] - y2tar], 2)
        ss1[i] = k2 / (1 + disttar1[i])
    
    for i in range(u2n):
        dist2[i] = np.linalg.norm([a2c[i] - a2d[i], a2c[i + u2n] - a2d[i + u2n]], 2)
        energy2[i] = k1 * V**2 * dist2[i]
        disttar2[i] = np.linalg.norm([a2d[i] - x1tar, a2d[i + u2n] - y1tar], 2)
        ss2[i] = k2 / (1 + disttar2[i])
    
    f1 = 1
    f2 = 0

    if f1 == 0 and f2 == 0:
        w11 = 0.5
        w12 = 1 - w11
        w21 = 0.5
        w22 = 1 - w11
    elif f1 == 0 and f2 == 1:
        w11 = 0.5
        w12 = 1 - w11
        w21 = 1.0
        w22 = 1 - w11
    elif f1 == 0 and f2 == 2:
        w11 = 0.5
        w12 = 1 - w11
        w21 = 0.5 + 0.5 * np.random.rand(1)
        w22 = 1 - w11
    elif f1 == 1 and f2 == 0:
        w11 = 1.0
        w12 = 1 - w11
        w21 = 0.5
        w22 = 1 - w11
    elif f1 == 1 and f2 == 1:
        w11 = 1.0
        w12 = 1 - w11
        w21 = 1.0
        w22 = 1 - w11
    elif f1 == 1 and f2 == 2:
        w11 = 1.0
        w12 = 1 - w11
        w21 = 0.5 + 0.5 * np.random.rand(1)
        w22 = 1 - w11
    elif f1 == 2 and f2 == 0:
        w11 = 0.5 + 0.5 * np.random.rand(1)
        w12 = 1 - w11
        w21 = 0.5
        w22 = 1 - w11
    else:
        w11 = 0.5 + 0.5 * np.random.rand(1)
        w12 = 1 - w11
        w21 = 0.5 + 0.5 * np.random.rand(1)
        w22 = 1 - w11

    # Calculate performance index for group 1
    p1 = w11 * ss1 + w12 * (1 / (1 + energy1))
    
    # Calculate performance index for group 2
    p2 = w21 * ss2 + w22 * (1 / (1 + energy2))
    
    z = np.concatenate((p1, p2))
    
    return z
