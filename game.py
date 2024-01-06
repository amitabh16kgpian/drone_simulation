import pygame
import sys
from uavdynamics import uavdynamics
import numpy as np
# Initialize Pygame
import random
import numpy as np 

import math

def drawDrone(color,drone_state):
    pygame.draw.circle(screen, color, (int(drone_state[0]), int(drone_state[1])), 21)  # Drone representation
    pygame.draw.circle(screen, white, (int(drone_state[0]), int(drone_state[1])), 20)  # Drone representation
    pygame.draw.line(screen, color, (int(drone_state[0]) - 3, int(drone_state[1])), (int(drone_state[0]) + 3, int(drone_state[1])), 1)
    pygame.draw.line(screen, color, (int(drone_state[0]), int(drone_state[1]) - 3), (int(drone_state[0]), int(drone_state[1]) + 3), 1)
    
def getCoord(drone_state,target):
    stheta=list(np.random.uniform(-1,1,10))
    ctheta=[(np.random.choice([-1, 1]))*math.sqrt(1-pow(t,2)) for t in stheta] 
    mdist=1000000
    radius=10
    mcoord=[]
    for i in range(len(stheta)):
        coord=np.array((drone_state[0]+radius*ctheta[i],drone_state[1]+radius*stheta[i]))
        dist=np.linalg.norm(coord - np.array(target))
        if dist<mdist:
            mdist=dist
            mcoord=list(coord)
    return mcoord
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Drone Simulation')

# Colors
white = (255, 255, 255)
red = (0, 255, 0)
drone1 = (0, 0, 255)
drone2 = (0, 200, 255)
num_d=10

# Initial conditions
initial_position = [0, 0]
initial_velocity = [1, 1]
target1 = [400, 300]
target2 = [500,200]

# Simulation parameters
simulation_time = 100
dt = 0.1

# Drone state
drone_state1 = [initial_position + initial_velocity]*num_d
drone_state2 = [initial_position + initial_velocity]*num_d

# Simulation loop
clock = pygame.time.Clock()
for t in np.arange(0, simulation_time, dt):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill(white)
    pygame.draw.circle(screen, red, target1, 5)  # Target waypoint
    pygame.draw.circle(screen, red, target2, 5)  # Target waypoint

    for d in range(num_d):
        mcoord=getCoord(drone_state1[d],target1)
        drone_state1[d]=uavdynamics(drone_state1[d]+mcoord)
        # drone_state = uavdynamics(drone_state + [target_waypoint[0], target_waypoint[1]])
        # Update display
        drawDrone(drone1,drone_state1[d])

    for d in range(num_d):
        
        mcoord=getCoord(drone_state2[d],target2)
        drone_state2[d]=uavdynamics(drone_state2[d]+mcoord)
        # drone_state = uavdynamics(drone_state + [target_waypoint[0], target_waypoint[1]])
        # Update display
        drawDrone(drone2,drone_state2[d])
    pygame.display.flip()
    # Control frame rate
    clock.tick(30)

pygame.quit()
sys.exit()


