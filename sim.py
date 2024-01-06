import matplotlib.pyplot as plt
from uavdynamics import uavdynamics
import numpy as np

# Initial conditions
initial_position = [0, 0]  # Initial position [x, y]
initial_velocity = [10, 5]  # Initial velocity [v_x, v_y]
target_waypoint = [100, 50]  # Target waypoint [x_ref, y_ref]

# Simulation parameters
simulation_time = 20  # Total simulation time in seconds
dt = 0.1  # Time step for simulation

# Lists to store trajectory data
positions = [initial_position]
velocities = [initial_velocity]

# Simulation loop
for t in np.arange(0, simulation_time, dt):
    current_state = positions[-1] + velocities[-1]
    current_state.extend(initial_velocity)  # Add initial velocity to the current state
    current_state.extend(target_waypoint)  # Add target waypoint to the current state
    new_state = uavdynamics(current_state)
    
    positions.append(new_state[:2])
    velocities.append(new_state[2:4])

# Extracting x and y coordinates for plotting
x_positions = [pos[0] for pos in positions]
y_positions = [pos[1] for pos in positions]

# Plotting the drone's trajectory
plt.plot(x_positions, y_positions, label='Drone Trajectory')
plt.scatter(target_waypoint[0], target_waypoint[1], color='red', label='Target Waypoint')
plt.title('Drone Trajectory Simulation')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()
