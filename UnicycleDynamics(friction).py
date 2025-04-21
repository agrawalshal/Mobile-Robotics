def unicycle_dynamics(state, control, dt=0.1):
    x, y, theta, friction = state
    v_cmd, omega_cmd = control

    # Apply friction
    v = v_cmd * (1 - friction)
    omega = omega_cmd * (1 - friction)

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    friction_next = friction

    return np.array([x_next, y_next, theta_next, friction_next])

def unicycle_measurement(state):
    return state[:2]
