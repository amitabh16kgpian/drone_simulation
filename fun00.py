# defining function to just calulate the distance between current and desired positon, desired and target position
# and energy requred to traverse that distance (main cheez to yahi hai)


import numpy as np

def fun00(a1c, a2c, a1d, a2d, x1tar, y1tar, x2tar, y2tar):
    k1 = 0.2 # Control parameter for cost function
    V = 20   # V = velociy
    k2 = 1000 # Another control parameter for cost function
    u1n = 1   
    u2n = 1   
    # a1c =  an array containing current position(x,y) of each element in group 1
    # a1d =  an array containing desired position(x,y) of each element in group 2
    # In this code we are just considering only one target for each group (i.e only 2 targert one for each group)
    # x1tar =  x-coordinate of target for group 1
    # y1tar =  y-coordinate of target for group 1
    
    # Just intializing the variables
    dist1 = np.zeros(u1n)    #dist1 = [0, 0, 0, 0, 0,] an array containing distance between current and desired position for each element in group 1
    energy1 = np.zeros(u1n)  #energy1 = energy required to cover the above distance
    disttar1 = np.zeros(u1n) #disttar1 = [0, 0, 0, 0, 0,] an array containing distance between desired and target position for each element in group 1
    ss1 = np.zeros(u1n)
    
    dist2 = np.zeros(u2n)
    energy2 = np.zeros(u2n)
    disttar2 = np.zeros(u2n)
    ss2 = np.zeros(u2n)
    
    for i in range(u1n): #uper defined variable ko calculate kar rahe hai for each element
        dist1[i] = np.linalg.norm([a1c[i] - a1d[i], a1c[i + u1n] - a1d[i + u1n]], 2)
        energy1[i] = k1 * V**2 * dist1[i]
        disttar1[i] = np.linalg.norm([a1d[i] - x1tar, a1d[i + u1n] - y1tar], 2)
        ss1[i] = k2 / (1 + disttar1[i])
    
    for i in range(u2n):
        dist2[i] = np.linalg.norm([a2c[i] - a2d[i], a2c[i + u2n] - a2d[i + u2n]], 2)
        energy2[i] = k1 * V**2 * dist2[i]
        disttar2[i] = np.linalg.norm([a2d[i] - x2tar, a2d[i + u2n] - y2tar], 2)
        ss2[i] = k2 / (1 + disttar2[i])
    
    #in this code we are calculating the performance index of entire group 1 and 2 individually considering the below case i.e f1=1 and f2 =0
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

    # Calculate performance index for entire group 1
    #While calculating preformance index, we can see that ss1 and energy1 are independent of values of  f1 and f2 or (w11,w12,w21,w22)
    #so externaly or the thing which is in our control is the value of f1 and f2 or (w11,w12,w21,w22) only 
    p1 = w11 * ss1 + w12 * (1 / (1 + energy1))
    
    # Calculate performance index for entire group 2
    p2 = w21 * ss2 + w22 * (1 / (1 + energy2))
    
    z = np.concatenate((p1, p2))
    
    return z
