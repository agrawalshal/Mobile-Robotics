def plot_all_results(true_states_ekf, estimated_states_ekf, controls_ekf, true_states_rsekf, estimated_states_rsekf, controls_rsekf, dt=0.1):

    # Create time vector
    time = np.arange(len(true_states_ekf)) * dt

    # Create a large figure with grid layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)

    # 1. Trajectory plot (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_trajectory(ax1, true_states_ekf, estimated_states_ekf,
                   true_states_rsekf, estimated_states_rsekf)

    # 2. Friction parameter estimation (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_friction(ax2, time, true_states_ekf, estimated_states_ekf,
                 true_states_rsekf, estimated_states_rsekf)

    # 3. Position error (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_position_error(ax3, time, true_states_ekf, true_states_rsekf)

    # 4. Linear velocity control (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    plot_linear_velocity(ax4, time, controls_ekf, controls_rsekf)

    # 5. Angular velocity control (bottom-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    plot_angular_velocity(ax5, time, controls_ekf, controls_rsekf)

    # 6. Additional metric: Distance to goal over time (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    plot_distance_to_goal(ax6, time, true_states_ekf, true_states_rsekf)

    # Add overall title and adjust layout
    fig.suptitle('Unicycle Control with EKF vs RS-EKF Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    return fig

def plot_trajectory(ax, true_states_ekf, estimated_states_ekf,
                   true_states_rsekf, estimated_states_rsekf):
    """Plot unicycle trajectories on given axis"""

    # Plot trajectories
    ax.plot(true_states_ekf[:, 0], true_states_ekf[:, 1], 'k-', linewidth=2, label='True Trajectory')
    ax.plot(estimated_states_ekf[:, 0], estimated_states_ekf[:, 1], 'b--', linewidth=2, label='EKF Estimate')
    ax.plot(estimated_states_rsekf[:, 0], estimated_states_rsekf[:, 1], 'r-', linewidth=2, label='RS-EKF Estimate')

    # Mark start and goal
    ax.plot(0, 0, 'go', markersize=10, label='Start')
    ax.plot(5, 5, 'r*', markersize=15, label='Goal')

    # Mark friction change point
    change_idx = 40
    ax.plot(true_states_ekf[change_idx, 0], true_states_ekf[change_idx, 1], 'mo',
            markersize=10, label='Friction Change')

    # Set axis limits with some padding
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)

    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_title('Unicycle Trajectory Comparison')
    ax.legend(loc='upper left')
    ax.grid(True)

def plot_friction(ax, time, true_states_ekf, estimated_states_ekf,
                 true_states_rsekf, estimated_states_rsekf):
    """Plot friction parameter estimation on given axis"""

    ax.plot(time, true_states_ekf[:, 3], 'k-', linewidth=2, label='True Friction')
    ax.plot(time, estimated_states_ekf[:, 3], 'b--', linewidth=2, label='EKF Estimate')
    ax.plot(time, estimated_states_rsekf[:, 3], 'r-', linewidth=2, label='RS-EKF Estimate')
    ax.axvline(x=4.0, color='g', linestyle='--', label='Friction Change')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Friction Coefficient')
    ax.set_title('Friction Parameter Estimation')
    ax.legend(loc='upper left')
    ax.grid(True)

def plot_position_error(ax, time, true_states_ekf, true_states_rsekf):
    """Plot position error on given axis"""

    goal = np.array([5.0, 5.0])

    ekf_pos_error = np.sqrt(np.sum((true_states_ekf[:, :2] - goal)**2, axis=1))
    rsekf_pos_error = np.sqrt(np.sum((true_states_rsekf[:, :2] - goal)**2, axis=1))

    ax.plot(time, ekf_pos_error, 'b--', linewidth=2, label='EKF Error')
    ax.plot(time, rsekf_pos_error, 'r-', linewidth=2, label='RS-EKF Error')
    ax.axvline(x=4.0, color='g', linestyle='--', label='Friction Change')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance to Goal [m]')
    ax.set_title('Position Error Comparison')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_linear_velocity(ax, time, controls_ekf, controls_rsekf):
    """Plot linear velocity control on given axis"""

    ax.plot(time, controls_ekf[:, 0], 'b--', linewidth=2, label='EKF Control')
    ax.plot(time, controls_rsekf[:, 0], 'r-', linewidth=2, label='RS-EKF Control')
    ax.axvline(x=4.0, color='g', linestyle='--', label='Friction Change')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Linear Velocity [m/s]')
    ax.set_title('Linear Velocity Control')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_angular_velocity(ax, time, controls_ekf, controls_rsekf):
    """Plot angular velocity control on given axis"""

    ax.plot(time, controls_ekf[:, 1], 'b--', linewidth=2, label='EKF Control')
    ax.plot(time, controls_rsekf[:, 1], 'r-', linewidth=2, label='RS-EKF Control')
    ax.axvline(x=4.0, color='g', linestyle='--', label='Friction Change')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity Control')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_distance_to_goal(ax, time, true_states_ekf, true_states_rsekf):
    """Plot distance to goal over time on given axis"""

    goal = np.array([5.0, 5.0])

    ekf_distance = np.sqrt(np.sum((true_states_ekf[:, :2] - goal)**2, axis=1))
    rsekf_distance = np.sqrt(np.sum((true_states_rsekf[:, :2] - goal)**2, axis=1))

    # Indicate the point where each approach reaches 0.5m from goal
    def time_to_reach(distances, threshold=0.5):
        for i, dist in enumerate(distances):
            if dist < threshold:
                return i
        return -1

    ekf_reach_idx = time_to_reach(ekf_distance)
    rsekf_reach_idx = time_to_reach(rsekf_distance)

    ax.plot(time, ekf_distance, 'b--', linewidth=2, label='EKF')
    ax.plot(time, rsekf_distance, 'r-', linewidth=2, label='RS-EKF')

    if ekf_reach_idx >= 0:
        ax.scatter(time[ekf_reach_idx], ekf_distance[ekf_reach_idx],
                  color='blue', s=100, marker='o',
                  label=f'EKF reaches goal at t={time[ekf_reach_idx]:.1f}s')

    if rsekf_reach_idx >= 0:
        ax.scatter(time[rsekf_reach_idx], rsekf_distance[rsekf_reach_idx],
                  color='red', s=100, marker='o',
                  label=f'RS-EKF reaches goal at t={time[rsekf_reach_idx]:.1f}s')

    ax.axvline(x=4.0, color='g', linestyle='--', label='Friction Change')
    ax.axhline(y=0.5, color='k', linestyle=':', label='Goal Threshold (0.5m)')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance to Goal [m]')
    ax.set_title('Goal Reaching Performance')
    ax.legend(loc='upper right')
    ax.grid(True)

def run_unicycle_comparison():
    print("Running standard EKF simulation (mu=0)...")
    true_states_ekf, estimated_states_ekf, controls_ekf = simulate_unicycle(
        num_steps=100, dt=0.1, mu=0.0, seed=42
    )

    mu_value = 0.05
    print(f"Running RS-EKF simulation with mu={mu_value}...")
    true_states_rsekf, estimated_states_rsekf, controls_rsekf = simulate_unicycle(
        num_steps=100, dt=0.1, mu=mu_value, seed=42
    )

    # Create and display the organized plots
    fig = plot_all_results(
        true_states_ekf, estimated_states_ekf, controls_ekf,
        true_states_rsekf, estimated_states_rsekf, controls_rsekf
    )

    time = np.arange(len(true_states_ekf)) * 0.1

    # Calculate statistics
    goal = np.array([5.0, 5.0])

    ekf_pos_error = np.sqrt(np.sum((true_states_ekf[:, :2] - goal)**2, axis=1))
    rsekf_pos_error = np.sqrt(np.sum((true_states_rsekf[:, :2] - goal)**2, axis=1))

    # Time to reach near goal (within 0.5m)
    def time_to_reach(errors, threshold=0.5):
        for i, err in enumerate(errors):
            if err < threshold:
                return i * 0.1
        return float('inf')

    ekf_time = time_to_reach(ekf_pos_error)
    rsekf_time = time_to_reach(rsekf_pos_error)

    # Compute MSE for pre and post friction change
    change_idx = 40

    pre_change_ekf_mse = np.mean(ekf_pos_error[:change_idx]**2)
    pre_change_rsekf_mse = np.mean(rsekf_pos_error[:change_idx]**2)

    post_change_ekf_mse = np.mean(ekf_pos_error[change_idx:]**2)
    post_change_rsekf_mse = np.mean(rsekf_pos_error[change_idx:]**2)

    total_ekf_mse = np.mean(ekf_pos_error**2)
    total_rsekf_mse = np.mean(rsekf_pos_error**2)

    # MSE for friction estimation
    friction_ekf_mse = np.mean((true_states_ekf[:, 3] - estimated_states_ekf[:, 3])**2)
    friction_rsekf_mse = np.mean((true_states_rsekf[:, 3] - estimated_states_rsekf[:, 3])**2)

    # Statistics
    print("\nPerformance Statistics:")
    print(f"Time to reach goal (within 0.5m):")
    print(f"  EKF: {ekf_time:.2f}s")
    print(f"  RS-EKF: {rsekf_time:.2f}s")

    print(f"\nBefore friction change:")
    print(f"  EKF Position MSE: {pre_change_ekf_mse:.6f}")
    print(f"  RS-EKF Position MSE: {pre_change_rsekf_mse:.6f}")
    if pre_change_ekf_mse > 0:
        pre_improvement = (1 - pre_change_rsekf_mse/pre_change_ekf_mse)*100
        print(f"  Improvement: {pre_improvement:.2f}%")

    print(f"\nAfter friction change:")
    print(f"  EKF Position MSE: {post_change_ekf_mse:.6f}")
    print(f"  RS-EKF Position MSE: {post_change_rsekf_mse:.6f}")
    if post_change_ekf_mse > 0:
        post_improvement = (1 - post_change_rsekf_mse/post_change_ekf_mse)*100
        print(f"  Improvement: {post_improvement:.2f}%")

    print(f"\nOverall position tracking:")
    print(f"  EKF MSE: {total_ekf_mse:.6f}")
    print(f"  RS-EKF MSE: {total_rsekf_mse:.6f}")
    if total_ekf_mse > 0:
        total_improvement = (1 - total_rsekf_mse/total_ekf_mse)*100
        print(f"  Improvement: {total_improvement:.2f}%")

    print(f"\nFriction parameter estimation:")
    print(f"  EKF MSE: {friction_ekf_mse:.6f}")
    print(f"  RS-EKF MSE: {friction_rsekf_mse:.6f}")
    if friction_ekf_mse > 0:
        friction_improvement = (1 - friction_rsekf_mse/friction_ekf_mse)*100
        print(f"  Improvement: {friction_improvement:.2f}%")

    plt.show()

    return (true_states_ekf, estimated_states_ekf, controls_ekf,
            true_states_rsekf, estimated_states_rsekf, controls_rsekf)
