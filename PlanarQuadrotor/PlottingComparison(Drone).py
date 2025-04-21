def plot_comparison(true_states, ekf_states, rsekf_states, ekf_controls, rsekf_controls, dt=0.05):
    time = np.arange(len(true_states)) * dt

    # Compute tracking error
    ekf_pos_error = np.sqrt((true_states[:, 0] - ekf_states[:, 0])**2 +
                           (true_states[:, 1] - ekf_states[:, 1])**2)
    rsekf_pos_error = np.sqrt((true_states[:, 0] - rsekf_states[:, 0])**2 +
                             (true_states[:, 1] - rsekf_states[:, 1])**2)

    # Create figure
    fig = plt.figure(figsize=(15, 12))

    # Position plot
    ax1 = fig.add_subplot(321)
    ax1.plot(true_states[:, 0], true_states[:, 1], 'k-', linewidth=2, label='True trajectory')
    ax1.plot(ekf_states[:, 0], ekf_states[:, 1], 'b--', label='EKF estimate')
    ax1.plot(rsekf_states[:, 0], rsekf_states[:, 1], 'r-.', label='RS-EKF estimate')
    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Y position (m)')
    ax1.set_title('Drone position')
    ax1.legend()
    ax1.grid(True)

    # Height plot
    ax2 = fig.add_subplot(322)
    ax2.plot(time, true_states[:, 1], 'k-', linewidth=2, label='True height')
    ax2.plot(time, ekf_states[:, 1], 'b--', label='EKF estimate')
    ax2.plot(time, rsekf_states[:, 1], 'r-.', label='RS-EKF estimate')
    ax2.axvline(x=2*dt, color='g', linestyle=':', label='Mass change')
    ax2.axvline(x=40*dt, color='g', linestyle=':')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Drone height')
    ax2.legend()
    ax2.grid(True)

    # Mass estimation plot
    ax3 = fig.add_subplot(323)
    ax3.plot(time, true_states[:, 6], 'k-', linewidth=2, label='True mass')
    ax3.plot(time, ekf_states[:, 6], 'b--', label='EKF estimate')
    ax3.plot(time, rsekf_states[:, 6], 'g--', label='RS-EKF estimate')
    ax3.axvline(x=2*dt, color='g', linestyle=':', label='Mass change')
    ax3.axvline(x=40*dt, color='g', linestyle=':')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass (kg)')
    ax3.set_title('Mass estimation')
    ax3.legend()
    ax3.grid(True)

    # Position error plot
    ax4 = fig.add_subplot(324)
    ax4.plot(time, ekf_pos_error, 'b--', label='EKF error')
    ax4.plot(time, rsekf_pos_error, 'r-.', label='RS-EKF error')
    ax4.axvline(x=2*dt, color='g', linestyle=':', label='Mass change')
    ax4.axvline(x=40*dt, color='g', linestyle=':')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position error (m)')
    ax4.set_title('Position tracking error')
    ax4.legend()
    ax4.grid(True)

    # Control inputs - EKF
    ax5 = fig.add_subplot(325)
    ax5.plot(time, ekf_controls[:, 0], 'b-', label='u1 (EKF)')
    ax5.plot(time, ekf_controls[:, 1], 'b--', label='u2 (EKF)')
    ax5.axvline(x=2*dt, color='g', linestyle=':', label='Mass change')
    ax5.axvline(x=40*dt, color='g', linestyle=':')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Thrust (N)')
    ax5.set_title('EKF control inputs')
    ax5.legend()
    ax5.grid(True)

    # Control inputs - RS-EKF
    ax6 = fig.add_subplot(326)
    ax6.plot(time, rsekf_controls[:, 0], 'r-', label='u1 (RS-EKF)')
    ax6.plot(time, rsekf_controls[:, 1], 'r--', label='u2 (RS-EKF)')
    ax6.axvline(x=2*dt, color='g', linestyle=':', label='Mass change')
    ax6.axvline(x=40*dt, color='g', linestyle=':')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Thrust (N)')
    ax6.set_title('RS-EKF control inputs')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()

# Run simulation with both filters
print("Running simulation with both EKF and RS-EKF...")
true_states, ekf_states, rsekf_states, ekf_controls, rsekf_controls = simulate_drone_with_both_filters(num_steps=100)

# Plot comparison
plot_comparison(true_states, ekf_states, rsekf_states, ekf_controls, rsekf_controls)

# Calculate performance metrics
ekf_mse = np.mean((true_states[:, :2] - ekf_states[:, :2])**2)
rsekf_mse = np.mean((true_states[:, :2] - rsekf_states[:, :2])**2)

print(f"\nPerformance metrics:")
print(f"EKF position MSE: {ekf_mse:.6f}")
print(f"RS-EKF position MSE: {rsekf_mse:.6f}")
print(f"Improvement: {(1 - rsekf_mse/ekf_mse)*100:.2f}%")

# Calculate trajectory cost for both controllers
def compute_trajectory_cost(states, controls, dt):
    total_cost = 0
    for t in range(len(controls)):
        state = states[t]
        control = controls[t]
        cost, _, _, _, _ = drone_cost(state, control, t)
        total_cost += cost * dt

    return total_cost / len(controls)  # Average cost

dt = 0.05
ekf_cost = compute_trajectory_cost(true_states, ekf_controls, dt)
rsekf_cost = compute_trajectory_cost(true_states, rsekf_controls, dt)
cost_improvement = (1 - rsekf_cost/ekf_cost) * 100

print("\nAverage cost along trajectory:")
print(f"Standard EKF cost: {ekf_cost:.4f}")
print(f"RS-EKF cost: {rsekf_cost:.4f}")
print(f"Cost improvement: {cost_improvement:.2f}%")  # Should be around 35% as reported in the paper

# Calculate responsiveness to mass change
ekf_mass_delay_1 = 0
rsekf_mass_delay_1 = 0
ekf_mass_delay_2 = 0
rsekf_mass_delay_2 = 0

mass_change_step_1 = 2
mass_change_step_2 = 40
mass_threshold = 3.0  # Consider detected when estimate exceeds this value

for t in range(mass_change_step_1, len(true_states)):
    if ekf_mass_delay_1 == 0 and ekf_states[t, 6] > mass_threshold:
        ekf_mass_delay_1 = t - mass_change_step_1
    if rsekf_mass_delay_1 == 0 and rsekf_states[t, 6] > mass_threshold:
        rsekf_mass_delay_1 = t - mass_change_step_1

for t in range(mass_change_step_2, len(true_states)):
    if ekf_mass_delay_2 == 0 and ekf_states[t, 6] < mass_threshold:
        ekf_mass_delay_2 = t - mass_change_step_2
    if rsekf_mass_delay_2 == 0 and rsekf_states[t, 6] < mass_threshold:
        rsekf_mass_delay_2 = t - mass_change_step_2

dt=0.05
print(f"\nMass change detection:")
print(f"EKF detected after {ekf_mass_delay_1} steps ({ekf_mass_delay_1*dt:.2f} seconds)")
print(f"RS-EKF detected after {rsekf_mass_delay_1} steps ({rsekf_mass_delay_1*dt:.2f} seconds)")
print(f"EKF detected after {ekf_mass_delay_2} steps ({ekf_mass_delay_2*dt:.2f} seconds)")
print(f"RS-EKF detected after {rsekf_mass_delay_2} steps ({rsekf_mass_delay_2*dt:.2f} seconds)")
