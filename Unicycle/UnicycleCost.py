def unicycle_cost(state, control, t, goal_state=np.array([5.0, 5.0, 0.0, 0.0])):
    if control is None:  # Terminal cost
        pos_cost = 100 * np.sum((state[:2] - goal_state[:2])**2)
        angle_cost = 10 * (state[2] - goal_state[2])**2

        cost = pos_cost + angle_cost

        cx = np.zeros(4)
        cx[0] = 200 * (state[0] - goal_state[0])
        cx[1] = 200 * (state[1] - goal_state[1])
        cx[2] = 20 * (state[2] - goal_state[2])

        cxx = np.diag([200, 200, 20, 0])

        return cost, cx, None, cxx, None
    else:

        pos_cost = 10 * np.sum((state[:2] - goal_state[:2])**2)
        angle_cost = 1 * (state[2] - goal_state[2])**2
        control_cost = 0.1 * (control[0]**2 + control[1]**2)

        cost = pos_cost + angle_cost + control_cost

        cx = np.zeros(4)
        cx[0] = 20 * (state[0] - goal_state[0])
        cx[1] = 20 * (state[1] - goal_state[1])
        cx[2] = 2 * (state[2] - goal_state[2])
        cx[3] = 5.0 * state[3]

        cu = np.array([
            0.2 * control[0],
            0.2 * control[1]
        ])

        cxx = np.diag([20, 20, 2, 5])
        cuu = 0.2 * np.eye(2)

        return cost, cx, cu, cxx, cuu

def simulate_unicycle(num_steps=100, dt=0.1, mu=0.05, seed=42):
    np.random.seed(seed)

    state_dim = 4        # [x, y, theta, friction]
    control_dim = 2      # [v, omega]
    measurement_dim = 2  # [x, y]
    horizon = 10         # Prediction horizon for DDP

    # Initialize true state: unicycle at origin, facing east, with low friction
    true_state = np.array([0.0, 0.0, 0.0, 0.1])

    # Initialize filter
    ekf = RiskSensitiveEKF(state_dim, measurement_dim, dt, mu)
    ekf.state_mean = np.array([0.0, 0.0, 0.0, 0.0])  # Initialize with zero friction estimate
    ekf.state_covariance = np.diag([0.01, 0.01, 0.01, 0.01])

    ekf.R = np.eye(measurement_dim) * 0.01  # Measurement noise
    ekf.Q = np.diag([0.01, 0.01, 0.01, 0.001])  # Process noise (lower for friction)

    ddp = DDP(
        dynamics_func=lambda x, u: unicycle_dynamics(x, u, dt),
        cost_func=unicycle_cost,
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        dt=dt
    )

    true_states = np.zeros((num_steps, state_dim))
    estimated_states = np.zeros((num_steps, state_dim))
    controls = np.zeros((num_steps, control_dim))

    u_sequence = np.zeros((horizon, control_dim))

    for t in range(num_steps):
        true_states[t] = true_state
        estimated_states[t] = ekf.state_mean

        # Change friction at specific time step
        if t == 40:
            true_state[3] = 0.5
            print(f"Step {t}: Friction increased to {true_state[3]}")

        try:
            x_sequence, u_sequence, K_sequence, vx, Vxx = ddp.optimize(ekf.state_mean, u_sequence)
            current_control = u_sequence[0]
        except Exception as e:
            print(f"DDP optimization failed: {e}")
            current_control = np.array([0.0, 0.0])

        current_control[0] = np.clip(current_control[0], -2.0, 2.0)
        current_control[1] = np.clip(current_control[1], -1.0, 1.0)

        controls[t] = current_control

        ekf.vx = vx
        ekf.Vxx = Vxx

        measurement_noise = np.random.multivariate_normal(np.zeros(measurement_dim), ekf.R)
        measurement = unicycle_measurement(true_state) + measurement_noise

        process_noise = np.random.multivariate_normal(np.zeros(state_dim), ekf.Q)
        true_state = unicycle_dynamics(true_state, current_control, dt) + process_noise

        ekf.predict(lambda x, u: unicycle_dynamics(x, u, dt), current_control)
        ekf.update(unicycle_measurement, measurement)

    return true_states, estimated_states, controls

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
