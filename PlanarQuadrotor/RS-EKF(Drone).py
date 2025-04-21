class RiskSensitiveEKF:
    def __init__(self, state_dim, measurement_dim, dt, mu=0.0):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        self.mu = mu

        # Initialize state estimate and covariance
        self.state_mean = np.zeros(state_dim)
        self.state_covariance = np.eye(state_dim)

        # Initialize noise covariances
        self.Q = np.eye(state_dim) * 0.01  # Process noise
        self.R = np.eye(measurement_dim) * 0.01  # Measurement noise

        # Value function derivatives (from DDP)
        self.vx = np.zeros(state_dim)
        self.Vxx = np.zeros((state_dim, state_dim))

    def predict(self, dynamics_func, control):
        """
        Prediction step of the filter
        """
        # Mean prediction using dynamics model
        x_prev = self.state_mean
        self.state_mean = dynamics_func(x_prev, control)

        # Compute Jacobian of dynamics model
        F = self._compute_dynamics_jacobian(dynamics_func, x_prev, control)

        # Covariance prediction
        self.state_covariance = F @ self.state_covariance @ F.T + self.Q

    def update(self, measurement_func, measurement):
        """
        Update step of the filter, incorporating risk sensitivity
        """
        # Predicted measurement
        y_pred = measurement_func(self.state_mean)

        # Compute Jacobian of measurement model
        H = self._compute_measurement_jacobian(measurement_func, self.state_mean)

        # Innovation covariance
        S = H @ self.state_covariance @ H.T + self.R

        # Kalman gain
        K = self.state_covariance @ H.T @ np.linalg.inv(S)

        # Standard EKF update
        delta_y = measurement - y_pred
        delta_x = K @ delta_y

        # Risk-sensitive modification (as described in the paper)
        if self.mu > 0:
            # Ensure (I - mu*P*Vxx) is positive definite
            I = np.eye(self.state_dim)
            mod_term = I - self.mu * self.state_covariance @ self.Vxx

            # Add a small regularization if needed
            eig_vals = np.linalg.eigvals(mod_term)
            if np.any(eig_vals <= 0):
                min_eig = np.min(eig_vals)
                if min_eig <= 0:
                    reg = 1e-6 - min_eig
                    mod_term += reg * I

            # Risk-sensitive update
            risk_term = self.mu * self.state_covariance @ self.vx
            delta_x_rs = np.linalg.solve(mod_term, delta_x + risk_term)

            # Update mean with risk-sensitive term
            self.state_mean += delta_x_rs
        else:
            # Standard EKF update
            self.state_mean += delta_x

        # Update covariance (standard EKF update)
        I = np.eye(self.state_dim)
        self.state_covariance = (I - K @ H) @ self.state_covariance

    def _compute_dynamics_jacobian(self, dynamics_func, state, control, eps=1e-6):
        """
        Compute the Jacobian of the dynamics function using finite differences
        """
        J = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.state_dim):
            state_plus = state.copy()
            state_plus[i] += eps

            f_plus = dynamics_func(state_plus, control)
            f = dynamics_func(state, control)

            J[:, i] = (f_plus - f) / eps

        return J

    def _compute_measurement_jacobian(self, measurement_func, state, eps=1e-6):
        """
        Compute the Jacobian of the measurement function using finite differences
        """
        J = np.zeros((self.measurement_dim, self.state_dim))

        for i in range(self.state_dim):
            state_plus = state.copy()
            state_plus[i] += eps

            h_plus = measurement_func(state_plus)
            h = measurement_func(state)

            J[:, i] = (h_plus - h) / eps

        return J
