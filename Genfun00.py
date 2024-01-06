#just for performance calculation considering that cooperation, coordination and coevolution 
import numpy as np

def genfun00(a1c, a2c, a1d, a2d, x1tar, y1tar, x2tar, y2tar):
    k1 = 0.2
    V = 20
    k2 = 1000
    
    u1n = int(0.5 * len(a1c))
    u2n = int(0.5 * len(a2c))
    
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
        disttar1[i] = np.linalg.norm([a1d[i] - x1tar[0], a1d[i + u1n] - y1tar[0]], 2)  #group-A ke 1st target
        ss1[i] = k2 / (1 + disttar1[i])

    for i in range(u2n):
        dist2[i] = np.linalg.norm([a2c[i] - a2d[i], a2c[i + u2n] - a2d[i + u2n]], 2)
        energy2[i] = k1 * V**2 * dist2[i]
        disttar2[i] = np.linalg.norm([a2d[i] - x2tar[0], a2d[i + u2n] - y2tar[0]], 2)  #group-B ke 1st target
        ss2[i] = k2 / (1 + disttar2[i])

    f1 = 1
    f2 = 1

    if (f1 == 0 and f2 == 0):
        w11 = 0.5
        w12 = 1 - w11
        w21 = 0.5
        w22 = 1 - w11
    elif (f1 == 0 and f2 == 1):
        w11 = 0.5
        w12 = 1 - w11
        w21 = 1.0
        w22 = 1 - w11
    elif (f1 == 0 and f2 == 2):
        w11 = 0.5
        w12 = 1 - w11
        w21 = 0.5 + 0.5 * np.random.rand()
        w22 = 1 - w11
    elif (f1 == 1 and f2 == 0):
        w11 = 1.0
        w12 = 1 - w11
        w21 = 0.5
        w22 = 1 - w11
    elif (f1 == 1 and f2 == 1):
        w11 = 1.0
        w12 = 1 - w11
        w21 = 1.0
        w22 = 1 - w11
    elif (f1 == 1 and f2 == 2):
        w11 = 1.0
        w12 = 1 - w11
        w21 = 0.5 + 0.5 * np.random.rand()
        w22 = 1 - w11
    elif (f1 == 2 and f2 == 0):
        w11 = 0.5 + 0.5 * np.random.rand()
        w12 = 1 - w11
        w21 = 0.5
        w22 = 1 - w11
    elif (f1 == 2 and f2 == 1):
        w11 = 0.5 + 0.5 * np.random.rand()
        w12 = 1 - w11
        w21 = 1.0
        w22 = 1 - w11
    else:
        w11 = 0.5 + 0.5 * np.random.rand()
        w12 = 1 - w11
        w21 = 0.5 + 0.5 * np.random.rand()
        w22 = 1 - w11

    ng1 = 1  #ng1 = no. of sub groups in group-1 ?
    p1 = np.zeros(ng1)
    
    for j in range(ng1):
        for i in range(u1n):
            p1[i] = w11 * ss1[i] + w12 * (1 / (1 + energy1[i]))

    ng2 = 1  #ng2 = no. of sub groups in group-2 ?
    p2 = np.zeros(ng2)
    
    for j in range(ng2):
        for i in range(u2n):
            p2[i] = w21 * ss2[i] + w22 * (1 / (1 + energy2[i]))

    z = np.concatenate((p1,p2))  
    
    return z 

