def drone_dynamics(state, control, dt=0.1):
    x, y, theta, v_x, v_y, omega, mass = state
    u1, u2 = control

    force_x = -(u1 + u2) * np.sin(theta)
    force_y = (u1 + u2) * np.cos(theta) - mass * 9.81
    torque = 0.5 * (u1 - u2)

    a_x = force_x / mass
    a_y = force_y / mass
    alpha = torque / mass

    v_x_next = v_x + a_x * dt
    v_y_next = v_y + a_y * dt
    omega_next = omega + alpha * dt

    x_next = x + v_x_next * dt
    y_next = y + v_y_next * dt
    theta_next = theta + omega_next * dt
    mass_next = mass

    return np.array([x_next, y_next, theta_next, v_x_next, v_y_next, omega_next, mass_next])

def drone_measurement(state):
    return state[:3]
