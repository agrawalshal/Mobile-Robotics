!apt-get install ffmpeg

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def create_simple_animation(true_states_ekf, estimated_states_ekf,
                           true_states_rsekf, estimated_states_rsekf,
                           dt=0.1):
    """
    Create a simple, appealing animation of the unicycle simulation
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Set up plot limits with padding
    max_x = max(np.max(true_states_ekf[:, 0]), np.max(true_states_rsekf[:, 0])) + 1
    max_y = max(np.max(true_states_ekf[:, 1]), np.max(true_states_rsekf[:, 1])) + 1
    min_x = min(np.min(true_states_ekf[:, 0]), np.min(true_states_rsekf[:, 0])) - 1
    min_y = min(np.min(true_states_ekf[:, 1]), np.min(true_states_rsekf[:, 1])) - 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Mark goal with a star
    goal = plt.scatter([5], [5], s=300, marker='*', color='gold', edgecolor='black', label='Goal', zorder=5)

    # Text for time and friction state
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    friction_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)

    # Initialize plot elements with empty data
    true_traj_ekf, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='True Path (EKF)')
    #est_traj_ekf, = ax.plot([], [], 'b--', linewidth=1.5, alpha=0.5, label='Est. Path (EKF)')
    true_traj_rsekf, = ax.plot([], [], 'r-', linewidth=2, alpha=0.6, label='True Path (RS-EKF)')
    #est_traj_rsekf, = ax.plot([], [], 'r--', linewidth=1.5, alpha=0.5, label='Est. Path (RS-EKF)')

    # Current positions
    true_pos_ekf = ax.scatter([], [], s=120, color='blue', alpha=0.8, label='True Pos (EKF)', zorder=10)
    #est_pos_ekf = ax.scatter([], [], s=80, color='blue', alpha=0.5, marker='o', label='Est. Pos (EKF)', zorder=9)
    true_pos_rsekf = ax.scatter([], [], s=120, color='red', alpha=0.8, label='True Pos (RS-EKF)', zorder=10)
    #est_pos_rsekf = ax.scatter([], [], s=80, color='red', alpha=0.5, marker='o', label='Est. Pos (RS-EKF)', zorder=9)

    # Add a vertical line to mark the friction change
    friction_line = ax.axvline(x=40*dt, color='green', linestyle='--', alpha=0.0, linewidth=2)

    # Set labels and title with larger font
    ax.set_xlabel('X Position [m]', fontsize=12)
    ax.set_ylabel('Y Position [m]', fontsize=12)
    ax.set_title('Unicycle Navigation: EKF vs. Risk-Sensitive EKF', fontsize=14)

    # Set legend at top left with transparency
    ax.legend(loc='lower right', fontsize=10, framealpha=0.7)

    # Add grid with light color
    ax.grid(True, alpha=0.3, linestyle='--')

    # Style the axes
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Number of frames (steps in the simulation)
    num_frames = len(true_states_ekf)

    def init():
        # Initialize with empty data
        true_traj_ekf.set_data([], [])
        #est_traj_ekf.set_data([], [])
        true_traj_rsekf.set_data([], [])
        #est_traj_rsekf.set_data([], [])

        # Set initial positions off-screen
        true_pos_ekf.set_offsets(np.array([[-100, -100]]))
        #est_pos_ekf.set_offsets(np.array([[-100, -100]]))
        true_pos_rsekf.set_offsets(np.array([[-100, -100]]))
        #est_pos_rsekf.set_offsets(np.array([[-100, -100]]))

        # Set initial text
        time_text.set_text('')
        friction_text.set_text('')

        # Hide the friction change line initially
        friction_line.set_alpha(0.0)

        return (true_traj_ekf, true_traj_rsekf,
                true_pos_ekf, true_pos_rsekf,
                time_text, friction_text, friction_line)

    def update(frame):
        # Update trajectory lines
        true_traj_ekf.set_data(true_states_ekf[:frame+1, 0], true_states_ekf[:frame+1, 1])
        #est_traj_ekf.set_data(estimated_states_ekf[:frame+1, 0], estimated_states_ekf[:frame+1, 1])
        true_traj_rsekf.set_data(true_states_rsekf[:frame+1, 0], true_states_rsekf[:frame+1, 1])
        #est_traj_rsekf.set_data(estimated_states_rsekf[:frame+1, 0], estimated_states_rsekf[:frame+1, 1])

        # Update current positions
        true_pos_ekf.set_offsets(np.array([[true_states_ekf[frame, 0], true_states_ekf[frame, 1]]]))
        #est_pos_ekf.set_offsets(np.array([[estimated_states_ekf[frame, 0], estimated_states_ekf[frame, 1]]]))
        true_pos_rsekf.set_offsets(np.array([[true_states_rsekf[frame, 0], true_states_rsekf[frame, 1]]]))
        #est_pos_rsekf.set_offsets(np.array([[estimated_states_rsekf[frame, 0], estimated_states_rsekf[frame, 1]]]))

        # Update time and friction text with more detail
        time_text.set_text(f'Time: {frame * dt:.1f}s')
        friction_text.set_text(f'Friction: {true_states_ekf[frame, 3]:.2f}')

        # Show friction change line if we've passed that point
        if frame >= 40:  # friction change index
            friction_line.set_alpha(0.6)
            friction_text.set_color('green')
        else:
            friction_line.set_alpha(0.0)
            friction_text.set_color('black')

        return (true_traj_ekf, true_traj_rsekf,
                true_pos_ekf, true_pos_rsekf,
                time_text, friction_text, friction_line)

    # Create animation with slower frame rate
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        interval=100, blit=True)  # Slower interval (100ms)

    plt.close()  # Prevent display of static plot

    return anim

# Function to display animation in Colab using HTML5
def display_animation_html5(anim):
    """Display animation in Colab using HTML5"""
    from IPython.display import HTML
    return HTML(anim.to_jshtml())

# Create the animation with the simplified style
anim = create_simple_animation(
    true_states_ekf,
    estimated_states_ekf,
    true_states_rsekf,
    estimated_states_rsekf
)

# Display the animation
display_animation_html5(anim)
