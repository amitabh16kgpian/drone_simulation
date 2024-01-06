import numpy as np
from Genfun00 import genfun00
from NearDist import NearDist
# Defining Main search space which is a square dimension 1000*1000
xmin, ymin = 0, 0
xmax, ymax = 1000, 1000

x1tar = [750, 500] #defining all (2-targets) the target coordinates for group 1
y1tar = [500, 520]
tar1 = len(x1tar)
x2tar = [500, 750] #defining all the target coordinates for group 2
y2tar = [520, 500]
tar2 = len(x2tar)
targets = tar1 + tar2  #targets = total number of targets

u1n, u2n = 10, 10  # no of agents in group 1 and 2
class UAV():
    def __init__(self):
        # St -Current State,    St+1 - next reference position, St+2 - next reference position,...............target
        #allocating the most initial x and y coordinates of group-1 drones
        self.x1 = np.random.uniform(xmin, xmax, u1n) #most initial positions [2.4, 4.5, 12.5, 54.1, 61.2, 35.4, 18.4, 19.4, 4.5, 8.5]
        self.y1 = np.random.uniform(ymin, ymax, u1n) #[2.4, 4.5, 12.5, 54.1, 61.2, 35.4, 18.4, 19.4, 4.5, 8.5]
        
        
        #allocating the x-ref and y-ref(just after most initial) of group-1 drones
        self.x1ref = np.random.uniform(xmin, xmax, u1n) #reference positions group-1 of drones [2.4, 4.5, 12.5, 54.1, 61.2, 35.4, 18.4, 19.4, 4.5, 8.5]
        self.y1ref = np.random.uniform(ymin, ymax, u1n) 

        #allocating the most initial x and y coordinates of group-2 drones
        self.x2 = np.random.uniform(xmin, xmax, u2n)  #most initial position
        self.y2 = np.random.uniform(ymin, ymax, u2n) 
        #allocating the x-ref and y-ref(just after most initial) of group-2 drones
        self.x2ref = np.random.uniform(xmin, xmax, u2n) #reference positions group-2 of drones
        self.y2ref = np.random.uniform(ymin, ymax, u2n) 
    def coevolution(self,a1c,a2c,a1d,a2d):
          
                # both of the groups are considered in the below code
                # Calling NearDist function here and storing the results in za, UAVtar1, UAVtar2, etc.
                # UAVtar1 means Group-1 ke drones group-1 ke kis target ko approach kar sakte hain
                # UAVtar2 means Group-1 ke drones group-2 ke kis target ko approach kar sakte hain
                # UAVtar3 means Group-2 ke drones group-2 ke kis target ko approach kar sakte hain
                # UAVtar4 means Group-2 ke drones group-1 ke kis target ko approach kar sakte hain
                #primary and secondary things are considered in the both code so order is different
                            #uds- yahan tak to kuchh nahi kiya bas chaaro target se closet drone nikal liya hai aur kuchh bhi nahi
                [za,UAVtar1,UAVtar2] = NearDist(a1c,a1c,a1d,a1d,x1tar,y1tar,x2tar,y2tar,targets); #Group-A = Group-1 and Group-B = Group-1
                [za1,UAVtar3,UAVtar4]= NearDist(a2c,a2c,a2d,a2d,x2tar,y2tar,x1tar,y1tar,targets); #Group-A = Group-2 and Group-B = Group-2
                #Dono groups ke paas dono group of targets ka information hai 
                #upar do baar nearest distance laga ke : group-1 ke drones ke liye dono group of targets me se optimal target kya rahega pta kar le rahe hai
                                                        # group-2 ke drones ke liye dono group of targets me se optimal target kya rahega pta kar le rahe hai
                # Coevolution 
                cev = 0  # cev=0 for coevolution
                # Two groups are evolving and two objective function
                # Two groups should share information
                # Group 1 targets ko dekh rahe hai abhi
                # It's a way of filtering elements from one array based on a condition specified by another array.
                        #Group-2 ke kaun se drones group-1 ke 2nd target ko approach karenge
                        #Upar jo Rishav socha tha usko yahan pe fir se leke aa rahe hai in more dynamic way          
                #seeing for group-1 target
                x22 = self.x2[UAVtar4 == 1] #group-2 ke jo jo drones group-1 ke 2nd target ko approach kar rahe hai unka coordinates utha ke x22 array me daal do
                y22 = self.y2[UAVtar4 == 1]
                #----------------------------change to uavtar2------------------------
                #make the code dynamic 
                PG1x = np.concatenate((self.x1, x22)) 
                PG1y = np.concatenate((self.y1, y22))
                
                PG1xy = np.concatenate((PG1x, PG1y))


                x22ref = self.x2ref[UAVtar4 == 1]
                y22ref = self.y2ref[UAVtar4 == 1]
                
                PG1xref = np.concatenate((self.x1ref, x22ref))
                PG1yref = np.concatenate((self.y1ref, y22ref))
                
                PG1xyref = np.concatenate((PG1xref, PG1yref))

                # Group 2 
                #seeing for group-2 target
                x11 = self.x1[UAVtar2 == 1] #group-1 ke jo jo drones group-2 ke 2nd target ko approach kar rahe hai unka coordinates utha ke x11 array me daal do
                y11 = self.y1[UAVtar2 == 1]
                # if x11.size > cev:
                PG2x = np.concatenate((self.x2[:10], x11))
                PG2y = np.concatenate((self.y2[:10], y11))
                # else:
                #     PG2x = self.x2[:10]
                #     PG2y = self.y2[:10]
                PG2xy = np.concatenate((PG2x, PG2y))


                x11ref = self.x1ref[UAVtar2 == 1]
                y11ref = self.y1ref[UAVtar2 == 1]
                # if x11.size > cev:
                PG2xref = np.concatenate((self.x2ref, x11ref))
                PG2yref = np.concatenate((self.y2ref, y11ref))
                # else:
                #     PG2xref = self.x2ref
                #     PG2yref = self.y2ref
                PG2xyref = np.concatenate((PG2xref, PG2yref))

                #here coevolution ends
                # due to the coevolution extra coordinates will be added so we need to (find best 10 out of it) calculate the new distance via a function

                #yahan se ab kuchh bhi kuchh bhi kar rahe hai
                # PG1f = Genfun00(PG1xy, PG2xy, PG1xyref, PG2xyref, x1tar, y1tar)   #Code me yahi tha but esse kuchh fruitfull nahi ho raha hai to comment kar diye. singnificance of PG1f??
                # PG2f = Genfun00(PG2xy, PG1xy, PG2xyref, PG1xyref, x2tar, y2tar)

                #ab humko best population ko figure out karna hai, uske liye hum dekhenge performance kis case me better hai aur fir chhat denge
                
                #this is basically selection of the fittest (i.e starting preparation for reproduction)
                        #pichhli bar fun00 function use kiye the usme input hamara ek ek point tha i.e value tha
                        #but Genfun00 me es baar pura array hi input kar de rahe hai
                PG1f = genfun00(PG1xy, PG2xy, PG1xyref, PG2xyref, x1tar, y1tar, x2tar, y2tar)   #agar group-1 ke drones group-1 ke 1st target ko approach karen and group-2 ke dornes group-2 ke 1st target ko approach karen to kya performance aaiga
                PG2f = genfun00(PG2xy, PG1xy, PG2xyref, PG1xyref,x1tar, y1tar, x2tar, y2tar)    #agar group-2 ke drones group-1 ke 1st target ko approach karen and group-1 ke drones group-2 ke 1st target ko approach karen to kya performance aaiga

                # self.x1 = PG1xref
                # self.y1 = PG1yref
                # self.x2 = PG2xref
                # self.y2 = PG2yref

        ####################################################################################################################
                # Sorting in descending order and storing the index
                indG1 = np.argsort(-PG1f)
                indG2 = np.argsort(-PG2f)

                #sorted arrays
                self.OrPG1f = PG1f[indG1]
                self.OrPG2f = PG2f[indG2]

                # TopPG1f = self.OrPG1f(1:10);
                TopPG1f = self.OrPG1f  

                # Reproduction starts
                TopPosG1xy = np.zeros((2, len(indG1)))

                for i in range(len(indG1)):
                    TopPosG1xy[0, i] = PG1xyref[indG1[i]]
                    TopPosG1xy[1, i] = PG1xyref[u1n + indG1[i]]

                xs1 = TopPosG1xy[0, :]
                ys1 = TopPosG1xy[1, :]

                z = TopPG1f

                # TopPosG2f = PG2xyref(:,1:10);
                # TopPG2f = self.OrPG2f(1:10);
                TopPG2f = self.OrPG2f 

                TopPosG2xy = np.zeros((2, len(indG2)))

                for i in range(len(indG2)):
                    TopPosG2xy[0, i] = PG2xyref[indG2[i]]
                    TopPosG2xy[1, i] = PG2xyref[u1n + indG2[i]]

                xs2 = TopPosG2xy[0, :]
                ys2 = TopPosG2xy[1, :]

                z1 = TopPG2f

                self.pv1 = np.zeros(len(indG1))
                self.pv2 = np.zeros(len(indG2))

                for j in range(len(indG1)):
                    self.pv1[j] = z[j]

                for j in range(len(indG2)):
                    self.pv2[j] = z1[j]

                # Group 2
                self.pv12 = np.zeros(u1n)

                for j in range(u1n):
                    self.pv12[j] = self.pv2[j]

                self.pv21 = np.zeros(u2n)

                for j in range(u1n, u1n + u2n):
                    self.pv21[j - u1n] = self.pv2[j]

                # Sorting in descending order
                x1s = np.sort(self.pv1)[::-1]
                x2s = np.sort(self.pv2)[::-1]

                # Getting the indices of the sorted arrays
                x1ps = np.argsort(-self.pv1)
                x2ps = np.argsort(-self.pv2)
                return indG1,indG2
    def evolve(self,x1ps,indG1):
        ctrack1 = len(x1ps)
        cntu1 = 0

        for i in range(2):     #not getting things from here onwards
            if x1ps[i] > u1n:
                for cntu1 in range(ctrack1):
                    if x1ps[ctrack1 - cntu1] < (u1n + 1):
                        self.x1ref[x1ps[ctrack1 - cntu1]] = self.x1[x1ps[i]]
                        self.y1ref[x1ps[ctrack1 - cntu1]] = self.y1[x1ps[i]]

                        self.x1[x1ps[ctrack1 - cntu1]] = self.x1ref[x1ps[ctrack1 - cntu1]]
                        self.y1[x1ps[ctrack1 - cntu1]] = self.y1ref[x1ps[ctrack1 - cntu1]]

                        x1ps[i] = x1ps[ctrack1 - cntu1]
                        break
            else:
                self.x1ref[x1ps[i]] = self.x1[x1ps[i]]
                self.y1ref[x1ps[i]] = self.y1[x1ps[i]]

        mc1 = 0
        x1b=['0']*len(self.x1)
        y1b=['0']*len(self.y1)
        x1rps=['0']*len(self.x1)
        x1bc=['0']*len(self.x1)
        y1bc=['0']*len(self.y1)
        for i in range(len(indG1)):
            if x1ps[i] < (u1n + 1):
                mc1 += 1
                x1b[mc1, :] = np.array(list(bin(int(round(self.x1[x1ps[i]])))[2:].zfill(10)))
                y1b[mc1, :] = np.array(list(bin(int(round(self.y1[x1ps[i]])))[2:].zfill(10)))
                x1rps[mc1] = x1ps[i]
                x1bc[mc1, :] = x1b[mc1, :]
                y1bc[mc1, :] = y1b[mc1, :]

        for i in range(2, 6):
            mtx = np.random.randint(1, 10)
            j = i
            if x1b[j, mtx] == '0':
                x1b[j, mtx] = '1'
            else:
                x1b[j, mtx] = '0'
            mty = np.random.randint(1, 10)
            if y1b[j, mty] == '0':
                y1b[j, mty] = '1'
            else:
                y1b[j, mty] = '0'
            self.x1ref[x1rps[i]] = int(''.join(x1b[j, :]), 2)
            self.y1ref[x1rps[i]] = int(''.join(y1b[j, :]), 2)

        # Crossover
        for i in range(6, 9, 2):
            ctx = np.random.randint(1, 9)
            for j in range(1, ctx + 1):
                x1b[i, j] = x1bc[i + 1, j]
                x1b[i + 1, j] = x1bc[i, j]
            cty = np.random.randint(1, 9)
            for j in range(1, cty + 1):
                y1b[i, j] = y1bc[i + 1, j]
                y1b[i + 1, j] = y1bc[i, j]

        for i in range(6, 10):
            self.x1ref[x1rps[i]] = int(''.join(x1b[i, :]), 2)
            self.y1ref[x1rps[i]] = int(''.join(y1b[i, :]), 2)
    def evolve2(self,x2ps,indG2):
        cntu2 = 0
        for i in range(2):
            if x2ps[i] > u2n:
                cntu2 += 1
                self.x2ref[cntu2 - 1] = self.x2[x2ps[i]]
                self.y2ref[cntu2 - 1] = self.y2[x2ps[i]]
            else:
                self.x2ref[x2ps[i]] = self.x2[x2ps[i]]
                self.y2ref[x2ps[i]] = self.y2[x2ps[i]]

        mc2 = 0
        x2b=['0']*len(self.x1)
        y2b=['0']*len(self.y1)
        x2rps=['0']*len(self.x1)
        x2bc=['0']*len(self.x1)
        y2bc=['0']*len(self.y1)
        # mutation for group 2
        for i in range(len(indG2)):
            if x2ps[i] < (u2n + 1):
                mc2 += 1
                x2b[mc2, :] = np.array(list(bin(int(round(self.x2[x2ps[i]])))[2:].zfill(10)))
                y2b[mc2, :] = np.array(list(bin(int(round(self.y2[x2ps[i]])))[2:].zfill(10)))
                x2rps[mc2] = x2ps[i]
                x2bc[mc2, :] = x2b[mc2, :]
                y2bc[mc2, :] = y2b[mc2, :]

        # Mutation for Group 2, binary conversion
        for i in range(2, 6):
            mtx = np.random.randint(1, 10)
            if x2b[i, mtx] == '0':
                x2b[i, mtx] = '1'
            else:
                x2b[i, mtx] = '0'
            mty = np.random.randint(1, 10)
            if y2b[i, mty] == '0':
                y2b[i, mty] = '1'
            else:
                y2b[i, mty] = '0'
            self.x2ref[x2rps[i]] = int(''.join(x2b[i, :]), 2)
            self.y2ref[x2rps[i]] = int(''.join(y2b[i, :]), 2)

        # Crossover for Group 2
        for i in range(6, 9, 2):
            ctx = np.random.randint(1, 9)
            for j in range(1, ctx + 1):
                x2b[i, j] = x2bc[i + 1, j]
                x2b[i + 1, j] = x2bc[i, j]
            cty = np.random.randint(1, 9)
            for j in range(1, cty + 1):
                y2b[i, j] = y2bc[i + 1, j]
                y2b[i + 1, j] = y2bc[i, j]

        for i in range(6, 10):
            self.x2ref[x2rps[i]] = int(''.join(x2b[i, :]), 2)
            self.y2ref[x2rps[i]] = int(''.join(y2b[i, :]), 2)  
        
    def fixInfeasible(self,un,xref,yref):
        for i in range(un):
            if xref[i] > xmax:
                xref[i] = (xmax - xmin) * np.random.rand() + xmin

            if xref[i] < xmin:
                xref[i] = (xmax - xmin) * np.random.rand() + xmin

            if yref[i] > ymax:
                yref[i] = (ymax - ymin) * np.random.rand() + ymin

            if yref[i] < ymin:
                yref[i] = (ymax - ymin) * np.random.rand() + ymin

    def solve(self):
        
        for drone_idx in range(10): #as coordinates of each drone are evolving, we need to access each drone
            # initializing current and reference position of drones in group 1 and group 2
            maxfit1=[]*500
            maxfit2=[]*500
            for iter in range(500): #iteration to reach the target
                #for group-1 considering both i.e group-1 target and group-2 targets also
                '''
                disttar1 = np.linalg.norm([self.x1ref - x1tar[0], self.y1ref - y1tar[0]], axis=0)#for target 1 #saare drones ka targert 1 se distance kitna hai
                disttar12 = np.linalg.norm([self.x1ref - x2tar[1], self.y1ref - y2tar[1]], axis=0)  #for target 2 #saare drones ka group-2 ke target 2 se distance kitna hai
                            #uds-group 1 ke drones jaan bujh ke group-1 ke 2nd target ko nhi dekh rahe hai jo ki perfect hai kyunki aapas me coordinate karna hai 
                disttar1 = np.linalg.norm([self.x1ref - x2tar[0], self.y1ref - y2tar[0]], axis=0)   #change the name of the variable
                disttar12 = np.linalg.norm([self.x1ref - x2tar[1], self.y1ref - y2tar[1]], axis=0)  #change the name of the variable
                        
                #target can be any number so make it generic 
                #for group-2
                disttar2 = np.linalg.norm([self.x2ref - x2tar[0], self.y2ref - y2tar[0]], axis=0)   #change the name of the variable
                disttar21 = np.linalg.norm([self.x2ref - x1tar[1], self.y2ref - y1tar[1]], axis=0)  #change the name of the variable
                            #uds-group 2 ke drones jaan bujh ke group-2 ke 2nd target ko nhi dekh rahe hai
                            #cosidering the secondary thing
                disttar2 = np.linalg.norm([self.x2ref - x1tar[0], self.y2ref - y1tar[0]], axis=0)  #edited code
                disttar21 = np.linalg.norm([self.x2ref - x1tar[1], self.y2ref - y1tar[1]], axis=0) #edited code in meet

                #make it generic code from 46 to 57
                #for group-1
                tar1dist = np.min(disttar1)
                indmin1 = np.argmin(disttar1) #group-1 ka drone number which is closest to 1st target of group-1

                tar12dist = np.min(disttar12)
                indmin12 = np.argmin(disttar12) #group-1 ka drone number which is closest to 2nd target of group-2

                            #in this case also consider that secondary thing and make the code generic 
            
                #for group-2
                tar2dist = np.min(disttar2)
                indmin2 = np.argmin(disttar2) #group-2 ka drone number which is closest to 1st target of group-2

                tar21dist = np.min(disttar21)
                indmin21 = np.argmin(disttar21) #group-2 ka drone number which is closest to 2nd target of group-1
                '''
                            #uds - till now we have covered all the 4 targets i.e drones which is closet to these 4 targets
                            #in this case also consider that secondary thing and make the code generic 
                # Group 1 GA (genetic algorithm karwana hai to kuchh na kuchh change karna hi padega na, to bas kuchh bhi kar rahe hai bas jo man me aaye)
                #-------------------------initial positions-------------------------------------

                a1c = np.concatenate((self.x1, self.y1))   #creating a1c using self.x1 and self.y1 as a1c = [x, x,x, x,x ,y,y,y,y,y]
                a2c = np.concatenate((self.x2, self.y2))
                
                #-------------------------Desired positions-------------------------------------
                a1d = np.concatenate((self.x1ref, self.y1ref)) #reference position is same as desired position 
                a2d = np.concatenate((self.x2ref, self.y2ref)) 
                indG1,indG2=self.coevolution(a1c,a2c,a1d,a2d)
                #preparing for cross over (so doing selection)
                x1ps = indG1 
                x2ps = indG2
                #Group 1 GA starts
                self.evolve(x1ps)
                # Group 2 GA starts
                self.evolve2(x2ps)

                maxfit1[iter] = self.OrPG1f[0]
                maxfit2[iter] = self.OrPG2f[0]
                # ------Group 2 GA ends---------------
                # Coevolution part
        # again evalute with new assigned coordinate positions
                # Calculate distances between agents in group 1 and group 2
                dist12n = np.zeros((u1n, u2n))
                dist21n = np.zeros((u2n, u1n))

                for i in range(u1n):
                    for j in range(u2n):
                        dist12n[i, j] = np.linalg.norm([self.x1ref[i] - self.x2ref[j], self.y1ref[i] - self.y2ref[j]], 2)

                for i in range(u2n):
                    for j in range(u1n):
                        dist21n[i, j] = np.linalg.norm([self.x2ref[i] - self.x1ref[j], self.y2ref[i] - self.y1ref[j]], 2)

                x1m = np.zeros((u1n, u2n))
                y1m = np.zeros((u1n, u2n))
                cnxt1 = np.zeros(u1n)
                sst12 = np.zeros((u1n, u2n))

                for i in range(u1n):
                    for j in range(u2n):
                        if dist12n[i, j] < rs:
                            x1m[i, j] = self.x2ref[j]
                            y1m[i, j] = self.y2ref[j]
                            sst12[i, j] = self.pv21[j]
                            cnxt1[i] += 1

                if iter > 50:
                    for i in range(u1n):
                        if cnxt1[i] > 0:
                            tx1, tx1c = np.sort(sst12[i, :], kind='heapsort')[::-1], np.argsort(sst12[i, :], kind='heapsort')[::-1]
                            scf1 = self.pv1[i] / (self.pv1[i] + tx1[0])
                            if self.pv1[i] < tx1[0]:
                                self.x1ref[i] = scf1 * self.x1ref[i] + (1 - scf1) * x1m[i, tx1c[0]]
                                self.y1ref[i] = scf1 * self.y1ref[i] + (1 - scf1) * y1m[i, tx1c[0]]

                # Clear temporary variables
                del tx1
                del tx1c

                # For the second group
                x2m = np.zeros((u2n, u1n))
                y2m = np.zeros((u2n, u1n))
                cnxt2 = np.zeros(u2n)
                sst21 = np.zeros((u2n, u1n))

                for i in range(u2n):
                    for j in range(u1n):
                        if dist21n[i, j] < rs:
                            x2m[i, j] = self.x1ref[j]
                            y2m[i, j] = self.y1ref[j]
                            sst21[i, j] = self.pv12[j]
                            cnxt2[i] += 1
                if iter > 50:
                    for i in range(u2n):
                        if cnxt2[i] > 0:
                            tx2, tx2c = np.sort(sst21[i, :], kind='heapsort')[::-1], np.argsort(sst21[i, :], kind='heapsort')[::-1]
                            scf2 = self.pv2[i] / (self.pv2[i] + tx2[0])
                            if self.pv2[i] < tx2[0]:
                                self.x2ref[i] = scf2 * self.x2ref[i] + (1 - scf2) * x2m[i, tx2c[0]]
                                self.y2ref[i] = scf2 * self.y2ref[i] + (1 - scf2) * y2m[i, tx2c[0]]

                # Clear temporary variables
                del tx2
                del tx2c

                # Infeasible solutions for Group 1
                self.fixInfeasible(u1n,self.x1ref,self.y1ref)
                for i in range(u1n):
                    if self.x1ref[i] > xmax:
                        self.x1ref[i] = (xmax - xmin) * np.random.rand() + xmin

                    if self.x1ref[i] < xmin:
                        self.x1ref[i] = (xmax - xmin) * np.random.rand() + xmin

                    if self.y1ref[i] > ymax:
                        self.y1ref[i] = (ymax - ymin) * np.random.rand() + ymin

                    if self.y1ref[i] < ymin:
                        self.y1ref[i] = (ymax - ymin) * np.random.rand() + ymin

                # Infeasible solutions for Group 2
                for i in range(u2n):
                    if self.x2ref[i] > xmax:
                        self.x2ref[i] = (xmax - xmin) * np.random.rand() + xmin

                    if self.x2ref[i] < xmin:
                        self.x2ref[i] = (xmax - xmin) * np.random.rand() + xmin

                    if self.y2ref[i] > ymax:
                        self.y2ref[i] = (ymax - ymin) * np.random.rand() + ymin

                    if self.y2ref[i] < ymin:
                        self.y2ref[i] = (ymax - ymin) * np.random.rand() + ymin

                # End of the main loop
            perf1[drone_idx] = np.mean(tar1dist)
            perf2[drone_idx] = np.mean(tar2dist)

# Calculating the mean performance
mean_perf1 = np.mean(perf1)
mean_perf2 = np.mean(perf2)

# Plotting the results
plt.figure(1)
plt.plot(tar1dist)
plt.figure(2)
plt.plot(tar2dist, 'r')

# Display the mean performance
print("Mean Performance Group 1:", mean_perf1)
print("Mean Performance Group 2:", mean_perf2)