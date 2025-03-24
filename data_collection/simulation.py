import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Register 3D projection
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# -------------------------------
# FIFA Standard Dimensions & Simulation Parameters
# -------------------------------

# FIFA standard pitch dimensions (in meters)
PITCH_LENGTH = 105.0  # Standard FIFA field length
PITCH_WIDTH = 68.0    # Standard FIFA field width
HALF_PITCH_LENGTH = PITCH_LENGTH / 2

# Goal dimensions
GOAL_WIDTH = 7.32     # Standard FIFA goal width
GOAL_HEIGHT = 2.44    # Standard FIFA goal height
GOAL_DEPTH = 2.0      # Added goal depth for visualization

# Penalty area and goal area dimensions
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.3
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.3
PENALTY_MARK_DISTANCE = 11.0
PENALTY_ARC_RADIUS = 9.15
CENTER_CIRCLE_RADIUS = 9.15
CORNER_ARC_RADIUS = 1.0

# Ball parameters
ball_position = np.array([0, 0, 0])  # Ball starting position
ball_speed = 27.0    # Ball speed [m/s]
theta = 15.0         # Elevation angle [deg]
phi = -25.0          # Azimuth angle [deg]

# Compute initial velocity components
vx = ball_speed * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
vy = ball_speed * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
vz = ball_speed * np.sin(np.radians(theta))
ball_velocity = np.array([vx, vy, vz])
states0 = np.concatenate((ball_position, ball_velocity))

# Simulation timing parameters
playback_speed = 0.1
tF = 3.0                              # Total simulation time [s]
fR = 30 / playback_speed              # Frame rate [fps]
dt = 1 / fR
t_eval = np.linspace(0, tF, int(tF * fR) + 1)

# -------------------------------
# Ball Dynamics and Event Functions
# -------------------------------

def ball_dynamics(t, states):
    """
    Calculate state vector derivatives
    State vector: [x, y, z, dx, dy, dz]
    """
    x, y, z, dx, dy, dz = states
    v = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Parameters
    m = 0.425      # Mass [kg]
    A = 0.0388     # Cross-sectional area [m^2]
    g = 9.8        # Gravitational acceleration [m/s^2]
    rho = 1.2      # Air density [kg/m^3]
    r = 0.111      # Ball radius [m]
    CM = 1.0       # Magnus coefficient
    CD = 0.275     # Drag coefficient

    # Ball angular velocity (rad/s)
    wx = 0
    wy = 0
    wz = 6 * 2 * np.pi  # 12*pi rad/s

    # Combined coefficients
    C1 = 0.5 * CD * rho * A
    C2 = 0.5 * CM * rho * A * r

    ddx = (-C1 * v * dx + C2 * (wy * dz - wz * dy)) / m
    ddy = (-C1 * v * dy + C2 * (wz * dx - wx * dz)) / m
    ddz = (-C1 * v * dz + C2 * (wx * dy - wy * dx) - m * g) / m

    return [dx, dy, dz, ddx, ddy, ddz]

def event_ball_end(t, states):
    """
    Terminate integration when ball crosses the endline with some margin
    """
    return (HALF_PITCH_LENGTH + 5) - states[0]
event_ball_end.terminal = True
event_ball_end.direction = 0

# Solve the ODE
sol = solve_ivp(ball_dynamics, [0, tF], states0, t_eval=t_eval, events=event_ball_end, rtol=1e-6)
t_sol = sol.t
states_sol = sol.y
x_sol, y_sol, z_sol = states_sol[0], states_sol[1], states_sol[2]

# -------------------------------
# Field Drawing Function
# -------------------------------

def plot_field(ax, camera_position):
    """
    Draw a full football pitch with FIFA standard dimensions on a 3D axis
    """
    # Draw the pitch boundary - full pitch
    # Goal lines
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH], 
            [-PITCH_WIDTH/2, PITCH_WIDTH/2], 
            [0, 0], 'k', linewidth=2)
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH], 
            [-PITCH_WIDTH/2, PITCH_WIDTH/2], 
            [0, 0], 'k', linewidth=2)
    
    # Sidelines
    ax.plot([-HALF_PITCH_LENGTH, HALF_PITCH_LENGTH], 
            [-PITCH_WIDTH/2, -PITCH_WIDTH/2], 
            [0, 0], 'k', linewidth=2)
    ax.plot([-HALF_PITCH_LENGTH, HALF_PITCH_LENGTH], 
            [PITCH_WIDTH/2, PITCH_WIDTH/2], 
            [0, 0], 'k', linewidth=2)
    
    # Center line (halfway line)
    ax.plot([0, 0], 
            [-PITCH_WIDTH/2, PITCH_WIDTH/2], 
            [0, 0], 'k', linewidth=2)
    
    # Center circle (full)
    theta = np.linspace(0, 2*np.pi, 60)
    center_circle_x = CENTER_CIRCLE_RADIUS * np.cos(theta)
    center_circle_y = CENTER_CIRCLE_RADIUS * np.sin(theta)
    ax.plot(center_circle_x, center_circle_y, np.zeros_like(theta), 'k', linewidth=1)
    
    # Center mark
    ax.scatter([0], [0], [0], c='k', s=20)
    
    # Corner arcs - draw all four corners
    # Top-left corner
    theta_corner_tl = np.linspace(0, np.pi/2, 10)
    corner_tl_x = -HALF_PITCH_LENGTH + CORNER_ARC_RADIUS * np.cos(theta_corner_tl)
    corner_tl_y = PITCH_WIDTH/2 - CORNER_ARC_RADIUS * np.sin(theta_corner_tl)
    ax.plot(corner_tl_x, corner_tl_y, np.zeros_like(theta_corner_tl), 'k', linewidth=1)
    
    # Bottom-left corner
    theta_corner_bl = np.linspace(-np.pi/2, 0, 10)
    corner_bl_x = -HALF_PITCH_LENGTH + CORNER_ARC_RADIUS * np.cos(theta_corner_bl)
    corner_bl_y = -PITCH_WIDTH/2 + CORNER_ARC_RADIUS * np.sin(theta_corner_bl)
    ax.plot(corner_bl_x, corner_bl_y, np.zeros_like(theta_corner_bl), 'k', linewidth=1)
    
    # Top-right corner
    theta_corner_tr = np.linspace(np.pi/2, np.pi, 10)
    corner_tr_x = HALF_PITCH_LENGTH - CORNER_ARC_RADIUS * np.cos(theta_corner_tr)
    corner_tr_y = PITCH_WIDTH/2 - CORNER_ARC_RADIUS * np.sin(theta_corner_tr)
    ax.plot(corner_tr_x, corner_tr_y, np.zeros_like(theta_corner_tr), 'k', linewidth=1)
    
    # Bottom-right corner
    theta_corner_br = np.linspace(np.pi, 3*np.pi/2, 10)
    corner_br_x = HALF_PITCH_LENGTH - CORNER_ARC_RADIUS * np.cos(theta_corner_br)
    corner_br_y = -PITCH_WIDTH/2 + CORNER_ARC_RADIUS * np.sin(theta_corner_br)
    ax.plot(corner_br_x, corner_br_y, np.zeros_like(theta_corner_br), 'k', linewidth=1)
    
    # Draw both penalty areas
    # Right penalty area
    pa_right_x = [HALF_PITCH_LENGTH, HALF_PITCH_LENGTH - PENALTY_AREA_LENGTH,
            HALF_PITCH_LENGTH - PENALTY_AREA_LENGTH, HALF_PITCH_LENGTH]
    pa_right_y = [-PENALTY_AREA_WIDTH/2, -PENALTY_AREA_WIDTH/2,
            PENALTY_AREA_WIDTH/2, PENALTY_AREA_WIDTH/2]
    pa_z = [0, 0, 0, 0]
    ax.plot(pa_right_x, pa_right_y, pa_z, 'k', linewidth=1)
    
    # Left penalty area
    pa_left_x = [-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH + PENALTY_AREA_LENGTH,
            -HALF_PITCH_LENGTH + PENALTY_AREA_LENGTH, -HALF_PITCH_LENGTH]
    pa_left_y = [-PENALTY_AREA_WIDTH/2, -PENALTY_AREA_WIDTH/2,
            PENALTY_AREA_WIDTH/2, PENALTY_AREA_WIDTH/2]
    ax.plot(pa_left_x, pa_left_y, pa_z, 'k', linewidth=1)
    
    # Draw both goal areas
    # Right goal area
    ga_right_x = [HALF_PITCH_LENGTH, HALF_PITCH_LENGTH - GOAL_AREA_LENGTH,
            HALF_PITCH_LENGTH - GOAL_AREA_LENGTH, HALF_PITCH_LENGTH]
    ga_right_y = [-GOAL_AREA_WIDTH/2, -GOAL_AREA_WIDTH/2,
            GOAL_AREA_WIDTH/2, GOAL_AREA_WIDTH/2]
    ga_z = [0, 0, 0, 0]
    ax.plot(ga_right_x, ga_right_y, ga_z, 'k', linewidth=1)
    
    # Left goal area
    ga_left_x = [-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH + GOAL_AREA_LENGTH,
            -HALF_PITCH_LENGTH + GOAL_AREA_LENGTH, -HALF_PITCH_LENGTH]
    ga_left_y = [-GOAL_AREA_WIDTH/2, -GOAL_AREA_WIDTH/2,
            GOAL_AREA_WIDTH/2, GOAL_AREA_WIDTH/2]
    ax.plot(ga_left_x, ga_left_y, ga_z, 'k', linewidth=1)
    
    # Draw both penalty marks
    ax.scatter([HALF_PITCH_LENGTH - PENALTY_MARK_DISTANCE], [0], [0], c='k', s=20)
    ax.scatter([-HALF_PITCH_LENGTH + PENALTY_MARK_DISTANCE], [0], [0], c='k', s=20)
    
    # Draw both penalty arcs - FIXED CALCULATION
    # The penalty arc should be drawn from the penalty mark, with the radius of PENALTY_ARC_RADIUS
    # Right penalty arc - start from angle that makes arc meet the penalty area line
    theta_arc_right = np.linspace(np.pi - 0.5, np.pi + 0.5, 20)  # Approximate angle range
    arc_right_x = HALF_PITCH_LENGTH - PENALTY_MARK_DISTANCE + PENALTY_ARC_RADIUS * np.cos(theta_arc_right)
    arc_right_y = PENALTY_ARC_RADIUS * np.sin(theta_arc_right)
    ax.plot(arc_right_x, arc_right_y, np.zeros_like(theta_arc_right), 'k', linewidth=1)
    
    # Left penalty arc
    theta_arc_left = np.linspace(-0.5, 0.5, 20)  # Approximate angle range
    arc_left_x = -HALF_PITCH_LENGTH + PENALTY_MARK_DISTANCE + PENALTY_ARC_RADIUS * np.cos(theta_arc_left)
    arc_left_y = PENALTY_ARC_RADIUS * np.sin(theta_arc_left)
    ax.plot(arc_left_x, arc_left_y, np.zeros_like(theta_arc_left), 'k', linewidth=1)
    
    # Draw right goal (same as your existing code)
    # Left post
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k', linewidth=3)
    
    # Right post
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k', linewidth=3)
    
    # Crossbar
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH],
            [-GOAL_WIDTH/2, GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k', linewidth=3)
    
    # Optional: Draw goal depth (back of the net)
    # Back corners
    ax.plot([HALF_PITCH_LENGTH + GOAL_DEPTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k--', linewidth=1)
    ax.plot([HALF_PITCH_LENGTH + GOAL_DEPTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k--', linewidth=1)
    
    # Top back
    ax.plot([HALF_PITCH_LENGTH + GOAL_DEPTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [-GOAL_WIDTH/2, GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k--', linewidth=1)
    
    # Side connections
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k--', linewidth=1)
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k--', linewidth=1)
    
    # Ground connections
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [0, 0],
            'k--', linewidth=1)
    ax.plot([HALF_PITCH_LENGTH, HALF_PITCH_LENGTH + GOAL_DEPTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [0, 0],
            'k--', linewidth=1)
    
    # Draw left goal (mirror of right goal)
    # Left post
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k', linewidth=3)
    
    # Right post
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k', linewidth=3)
    
    # Crossbar
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH],
            [-GOAL_WIDTH/2, GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k', linewidth=3)
    
    # Left goal depth
    ax.plot([-HALF_PITCH_LENGTH - GOAL_DEPTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k--', linewidth=1)
    ax.plot([-HALF_PITCH_LENGTH - GOAL_DEPTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [0, GOAL_HEIGHT],
            'k--', linewidth=1)
    
    # Top back
    ax.plot([-HALF_PITCH_LENGTH - GOAL_DEPTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [-GOAL_WIDTH/2, GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k--', linewidth=1)
    
    # Side connections
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k--', linewidth=1)
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [GOAL_HEIGHT, GOAL_HEIGHT],
            'k--', linewidth=1)
    
    # Ground connections
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [-GOAL_WIDTH/2, -GOAL_WIDTH/2],
            [0, 0],
            'k--', linewidth=1)
    ax.plot([-HALF_PITCH_LENGTH, -HALF_PITCH_LENGTH - GOAL_DEPTH],
            [GOAL_WIDTH/2, GOAL_WIDTH/2],
            [0, 0],
            'k--', linewidth=1)
    
    # Draw kick-off position
    ax.scatter([ball_position[0]], [ball_position[1]], [ball_position[2]], c='r', s=30)
    
    # Set axis limits AFTER all plotting is done
    ax.set_xlim([-HALF_PITCH_LENGTH - 30, HALF_PITCH_LENGTH + 30])
    ax.set_ylim([-PITCH_WIDTH/2 - 30, PITCH_WIDTH/2 + 30])
    ax.set_zlim([0, 20])  # Increased to show goal height properly
    
    # Force axis to respect the limits
    ax.autoscale(enable=False)
    
    # Set camera viewpoint AFTER limits are set
    cam = np.array(camera_position)
    elev = np.degrees(np.arctan2(cam[2], np.sqrt(cam[0]**2 + cam[1]**2)))
    azim = np.degrees(np.arctan2(cam[1], cam[0]))
    ax.view_init(elev=elev, azim=azim)

# -------------------------------
# Set up Figure and Subplots Layout for Top-Left View only
# -------------------------------

fig = plt.figure(figsize=(14, 10))
# Single plot with top-left view
ax1 = fig.add_subplot(111, projection='3d')

# Define camera position for top-left view
# Adjust the values for the top-left view
top_left_cam = [-80, 80, 100]  # Position coordinates for top-left view

# -------------------------------
# Animation Update Function
# -------------------------------

def update(frame):
    # Clear subplot
    ax1.cla()
    
    # Draw field with top-left camera angle
    plot_field(ax1, top_left_cam)
    
    # Draw ball trajectory up to current frame
    ax1.plot(x_sol[:frame + 1], y_sol[:frame + 1], z_sol[:frame + 1], 'r', linewidth=2)
    
    # Mark current ball position
    ax1.scatter(x_sol[frame], y_sol[frame], z_sol[frame], c='b', s=40)
    
    # Set labels and title
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_title(f'Free Kick Simulation - Time={t_sol[frame]:.3f} s')
    
    plt.tight_layout()

# -------------------------------
# Create and Save Animation
# -------------------------------

frames = len(t_sol)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt * 1000)

# Save animation as GIF (requires Pillow)
ani.save('free_kick_fifa_standard.gif', writer='pillow', fps=fR)

plt.show()