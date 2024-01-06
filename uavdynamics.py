# this code models how a UAV moves towards a specified waypoint by adjusting its velocity over time to 
# reach the desired path. It's a basic representation of UAV dynamics, and we can use this function to 
# simulate the UAV's movement given its initial conditions and a target waypoint.

#here the input is the current position coordinates,desired postion and current velocity
import numpy as np

def uavdynamics(xin):
    x1 = xin[0]  # UAV x position
    y1 = xin[1]  # UAV y position
    v1x = xin[2]  # UAV x velocity
    v1y = xin[3]  # UAV y velocity
    x1ref = xin[4]  # Reference position(coordinates of points where UAV wants to go) (waypoint) for UAV
    y1ref = xin[5]  #only the ref coordinates will be fixed rest current postion and current velocity will be changing

    dt = 0.1 
    Va = 20.0
    theta1 = np.arctan2((y1ref - y1), (x1ref - x1)) #theta = angle between the UAV's current position and the reference waypoint
    v1xref = Va * np.cos(theta1) #desired x and y velocities (v1xref and v1yref) for the UAV based on its current position and the reference waypoint
    v1yref = Va * np.sin(theta1)

    # Dynamics propagation
    v1xdot = -3 * v1x + 3 * v1xref   #v1xdot = dynamic(dynamic because this function will be running a loop, so each time i.e at different current position the v1x will different so acceleration will also be different) acceleration of the UAV 
    v1x = v1x + v1xdot * dt  
    v1ydot = -3 * v1y + 3 * v1yref   #v1ydot = dynamic acceleration of the UAV 
    v1y = v1y + v1ydot * dt
    x1dot = v1x
    x1 = x1 + x1dot * dt
    y1dot = v1y
    y1 = y1 + y1dot * dt
    xout = [x1, y1, v1x, v1y]       #this contains the current position and current velocity after dt time

    return xout



