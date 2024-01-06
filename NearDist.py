import numpy as np
from Genfun00 import Genfun00
#NearDist decide kar raha hai ki group-A and group-B ka har drone apne group wale targets me se kis target ko appraoch karega 

def NearDist(a1c, a2c, a1d, a2d, x1tar, y1tar, x2tar, y2tar, targets):
    t1 = x1tar.shape[1]
    t2 = x2tar.shape[1]
    k2 = 1000
    u1n = 10
    u2n = 10

    disttar1 = np.zeros((u1n, t1))
    disttar2 = np.zeros((u2n, t2))
    ss1 = np.zeros(u1n)
    ss2 = np.zeros(u1n)
    knn1 = np.zeros((u1n, t1))
    UAVtar1 = np.zeros(u1n)
    UAVtar2 = np.zeros(u1n)

    for k in range(t1):
        for i in range(u1n):
            disttar1[i, k] = np.linalg.norm([a1d[i] - x1tar[k], a1d[i + u1n] - y1tar[k]], 2)
            # ss1[i] = k2 / (1 + disttar1[i, k])

    for k in range(t2):
        for i in range(u1n):
            disttar2[i, k] = np.linalg.norm([a1d[i] - x2tar[k], a1d[i + u1n] - y2tar[k]], 2)
            # ss2[i] = k2 / (1 + disttar2[i, k])

    for i in range(u1n):
        ind1 = np.argmin(disttar1[i, :]) 
        ind2 = np.argmin(disttar2[i, :]) 
        abc1 = np.array([a1c[i], a1c[i + u1n]])
        abc2 = np.array([a1d[i], a1d[i + u1n]])
        abc3 = np.array([a2c[i], a2c[i + u1n]])
        abc4 = np.array([a2d[i], a2d[i + u1n]])
        UAVtar1[i] = ind1
        UAVtar2[i] = ind2
        knn1[i, :] = Genfun00(abc1, abc3, abc2, abc4, x1tar[ind1], y1tar[ind1], x2tar[ind2], y2tar[ind2])

    return knn1, UAVtar1, UAVtar2  #agar drones UAVtar1, UAVtar2 me jo drone accolated hai unko approach karega to performance kya hoga wo cheez knn1 me store ho raha hai
