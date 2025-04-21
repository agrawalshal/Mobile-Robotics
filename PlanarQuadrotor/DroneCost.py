def drone_cost(state, control, t, goal_state=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])):
    if control is None:  # Terminal cost
        # Higher weights for terminal cost
        pos_cost = 100 * ((state[0] - goal_state[0])**2 + (state[1] - goal_state[1])**2)
        angle_cost = 10 * (state[2] - goal_state[2])**2

        cost = pos_cost + angle_cost

        # Gradient of cost with respect to state
        cx = np.zeros(7)
        cx[0] = 200 * (state[0] - goal_state[0])  # Gradient for px
        cx[1] = 200 * (state[1] - goal_state[1])  # Gradient for py
        cx[2] = 20 * (state[2] - goal_state[2])   # Gradient for theta

        # Hessian of cost with respect to state
        cxx = np.diag([200, 200, 20, 0, 0, 0, 0])

        return cost, cx, None, cxx, None
    else:
        # Stage cost
        pos_cost = 100 * ((state[0] - goal_state[0])**2 + (state[1] - goal_state[1])**2)
        angle_cost = 10 * (state[2] - goal_state[2])**2
        velocity_cost = 0.01 * (state[3]**2 + state[4]**2 + state[5]**2)

        hover_thrust = state[6] * 9.81 / 2
        u_ref = np.array([hover_thrust, hover_thrust])
        control_cost = 0.1 * np.sum((control - u_ref)**2)

        cost = pos_cost + angle_cost + velocity_cost + control_cost

        # Gradient of cost with respect to state
        cx = np.zeros(7)
        cx[0] = 20 * (state[0] - goal_state[0])
        cx[1] = 20 * (state[1] - goal_state[1])
        cx[2] = 2 * (state[2] - goal_state[2])
        cx[3] = 0  
        cx[4] = 0  
        cx[5] = 0  
        cx[6] = 0  

        # Gradient of cost with respect to control
        cu = 0.2 * (control - u_ref)

        # Hessian of cost with respect to state
        cxx = np.diag([20, 20, 2, 0, 0, 0, 0])

        # Hessian of cost with respect to control
        cuu = 0.2 * np.eye(2)

        return cost, cx, cu, cxx, cuu

def simulate_drone_with_both_filters(num_steps=100, dt=0.05, mu_rs=4e-3, seed=42):
    np.random.seed(seed)

    state_dim = 7       # [x, y, theta, vx, vy, omega, mass]
    control_dim = 2     # [u1, u2]
    measurement_dim = 3 # [x, y, theta]
    horizon = 20        # Prediction horizon for DDP

    # Initialize true state: drone at origin with known mass initially
    true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])

    # Initialize standard EKF
    ekf = RiskSensitiveEKF(state_dim, measurement_dim, dt, mu=0.0)  # mu=0 gives standard EKF
    ekf.state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    ekf.state_covariance = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    ekf.R = np.eye(measurement_dim) * 0.0001  # Measurement noise
    ekf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.002])  # Process noise

    # Initialize RS-EKF (same initial conditions, but with risk sensitivity)
    rsekf = RiskSensitiveEKF(state_dim, measurement_dim, dt, mu=mu_rs)
    rsekf.state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    rsekf.state_covariance = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    rsekf.R = np.eye(measurement_dim) * 0.0001  # Same noise parameters as in research paper
    rsekf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.002])

    # Initialize DDPs for both filters
    ekf_ddp = DDP(
        dynamics_func=lambda x, u: drone_dynamics(x, u, dt),
        cost_func=drone_cost,
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        dt=dt
    )

    rsekf_ddp = DDP(
        dynamics_func=lambda x, u: drone_dynamics(x, u, dt),
        cost_func=drone_cost,
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        dt=dt
    )

    true_states = np.zeros((num_steps, state_dim))
    ekf_states = np.zeros((num_steps, state_dim))
    rsekf_states = np.zeros((num_steps, state_dim))
    ekf_controls = np.zeros((num_steps, control_dim))
    rsekf_controls = np.zeros((num_steps, control_dim))

    # Initial control sequences
    ekf_u_sequence = np.zeros((horizon, control_dim))
    rsekf_u_sequence = np.zeros((horizon, control_dim))

    # Main simulation loop
    for t in range(num_steps):
        true_states[t] = true_state
        ekf_states[t] = ekf.state_mean
        rsekf_states[t] = rsekf.state_mean

        # Change mass at specific time step to simulate picking up a load
        if t == 2:
            true_state[6] = 5.0  # Mass increases to 5kg
            print(f"Step {t}: Mass increased to {true_state[6]}")

        if t == 40:
            true_state[6] = 2.0  # Mass decreases from 5kg to 2kg
            print(f"Step {t}: Mass decreased to {true_state[6]}")

        measurement_noise = np.random.multivariate_normal(np.zeros(measurement_dim), ekf.R)
        measurement = drone_measurement(true_state) + measurement_noise

        # Optimize control for EKF
        try:
            ekf_x_sequence, ekf_u_sequence, ekf_K_sequence, ekf_vx, ekf_Vxx = ekf_ddp.optimize(
                ekf.state_mean, ekf_u_sequence
            )
            ekf_control = ekf_u_sequence[0]
        except Exception as e:
            print(f"EKF DDP optimization failed: {e}")
            # Fallback: hover control
            ekf_control = np.array([true_state[6] * 9.81 / 2, true_state[6] * 9.81 / 2])

        # Optimize control for RS-EKF
        try:
            rsekf_x_sequence, rsekf_u_sequence, rsekf_K_sequence, rsekf_vx, rsekf_Vxx = rsekf_ddp.optimize(
                rsekf.state_mean, rsekf_u_sequence
            )
            rsekf_control = rsekf_u_sequence[0]
        except Exception as e:
            print(f"RS-EKF DDP optimization failed: {e}")
            # Fallback: hover control
            rsekf_control = np.array([true_state[6] * 9.81 / 2, true_state[6] * 9.81 / 2])

        # Clip controls 
        ekf_control[0] = np.clip(ekf_control[0], 0.0, 30.0)
        ekf_control[1] = np.clip(ekf_control[1], 0.0, 30.0)
        rsekf_control[0] = np.clip(rsekf_control[0], 0.0, 30.0)
        rsekf_control[1] = np.clip(rsekf_control[1], 0.0, 30.0)

        ekf_controls[t] = ekf_control
        rsekf_controls[t] = rsekf_control

        control = rsekf_control

        ekf.vx = ekf_vx
        ekf.Vxx = ekf_Vxx
        rsekf.vx = rsekf_vx
        rsekf.Vxx = rsekf_Vxx

        process_noise = np.random.multivariate_normal(np.zeros(state_dim), ekf.Q)
        true_state = drone_dynamics(true_state, control, dt) + process_noise

        ekf.predict(lambda x, u: drone_dynamics(x, u, dt), ekf_control)
        ekf.update(drone_measurement, measurement)

        rsekf.predict(lambda x, u: drone_dynamics(x, u, dt), rsekf_control)
        rsekf.update(drone_measurement, measurement)

    return true_states, ekf_states, rsekf_states, ekf_controls, rsekf_controls
