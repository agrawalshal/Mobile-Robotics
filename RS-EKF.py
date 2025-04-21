class RiskSensitiveEKF:
    def __init__(self, state_dim, measurement_dim, dt=0.1, mu=0.05):

        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        self.mu = mu

        # State estimate and covariance
        self.state_mean = np.zeros(state_dim)
        self.state_covariance = np.eye(state_dim)

        # Value function derivatives
        self.vx = np.zeros(state_dim)
        self.Vxx = np.zeros((state_dim, state_dim))

    def predict(self, dynamics_func, control):

        fx = self.compute_dynamics_jacobian(dynamics_func, self.state_mean, control)

        # Predict state
        self.state_mean = dynamics_func(self.state_mean, control)

        # Predict covariance
        self.state_covariance = fx @ self.state_covariance @ fx.T + self.Q

        return self.state_mean.copy(), self.state_covariance.copy()

    def update(self, measurement_func, measurement):

        hx = self.compute_measurement_jacobian(measurement_func, self.state_mean)

        expected_z = measurement_func(self.state_mean)

        innovation = measurement - expected_z

        S = hx @ self.state_covariance @ hx.T + self.R

        K = self.state_covariance @ hx.T @ np.linalg.inv(S)

        delta_x = K @ innovation

        I = np.eye(self.state_dim)
        self.state_covariance = (I - K @ hx) @ self.state_covariance @ (I - K @ hx).T + K @ self.R @ K.T

        if self.mu == 0 or (np.all(self.vx == 0) and np.all(self.Vxx == 0)):
            self.state_mean = self.state_mean + delta_x
        else:
            P = self.state_covariance
            I = np.eye(self.state_dim)

            try:
                risk_matrix = I - self.mu * P @ self.Vxx
                risk_factor = np.linalg.solve(risk_matrix, delta_x + self.mu * P @ self.vx)
            except np.linalg.LinAlgError:
                reg = 1e-6
                reg_matrix = I - self.mu * P @ (self.Vxx - reg * I)
                try:
                    risk_factor = np.linalg.solve(reg_matrix, delta_x + self.mu * P @ self.vx)
                except:
                    risk_factor = delta_x
                    print("Warning: Using standard EKF update due to numerical issues")

            self.state_mean = self.state_mean + risk_factor

        return self.state_mean.copy(), self.state_covariance.copy()

    def compute_dynamics_jacobian(self, dynamics_func, state, control, eps=1e-6):
        fx = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.state_dim):
            state_plus = state.copy()
            state_plus[i] += eps

            fx[:, i] = (dynamics_func(state_plus, control) - dynamics_func(state, control)) / eps

        return fx

    def compute_measurement_jacobian(self, measurement_func, state, eps=1e-6):
        hx = np.zeros((self.measurement_dim, self.state_dim))

        for i in range(self.state_dim):
            state_plus = state.copy()
            state_plus[i] += eps

            hx[:, i] = (measurement_func(state_plus) - measurement_func(state)) / eps

        return hx
