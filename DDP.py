class DDP:
    def __init__(self, dynamics_func, cost_func, state_dim, control_dim, horizon, dt=0.1):

        self.dynamics_func = dynamics_func
        self.cost_func = cost_func
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.dt = dt

        # Regularization
        self.reg_min = 1e-6
        self.reg_max = 1e10
        self.reg_factor = 10
        self.reg = self.reg_min

        # Line search parameters
        self.alpha = 1.0
        self.alpha_min = 1e-4
        self.alpha_decay = 0.5

        # Convergence parameters
        self.max_iterations = 100
        self.tol = 1e-4

    def rollout(self, x0, u_sequence):
        x_sequence = np.zeros((self.horizon + 1, self.state_dim))
        x_sequence[0] = x0

        total_cost = 0

        for t in range(self.horizon):
            # Calculate cost
            cost, _, _, _, _ = self.cost_func(x_sequence[t], u_sequence[t], t)
            total_cost += cost

            # Propagate
            x_sequence[t+1] = self.dynamics_func(x_sequence[t], u_sequence[t])

        # Terminal cost
        terminal_cost, _, _, _, _ = self.cost_func(x_sequence[-1], None, self.horizon)
        total_cost += terminal_cost

        return x_sequence, total_cost

    def compute_derivatives(self, x, u, t):
        fx = np.zeros((self.state_dim, self.state_dim))
        fu = np.zeros((self.state_dim, self.control_dim))

        eps = 1e-6
        for i in range(self.state_dim):
            # Perturb state
            x_perturbed = x.copy()
            x_perturbed[i] += eps

            # Forward difference for fx
            fx[:, i] = (self.dynamics_func(x_perturbed, u) - self.dynamics_func(x, u)) / eps

        if u is not None:
            for i in range(self.control_dim):
                # Perturb control
                u_perturbed = u.copy()
                u_perturbed[i] += eps

                # Forward difference for fu
                fu[:, i] = (self.dynamics_func(x, u_perturbed) - self.dynamics_func(x, u)) / eps

        # Cost derivatives from the cost function
        cost, cx, cu, cxx, cuu = self.cost_func(x, u, t)

        return fx, fu, cost, cx, cu, cxx, cuu

    def backward_pass(self, x_sequence, u_sequence):
        # Value function derivatives 
        cost, Vx, _, Vxx, _ = self.cost_func(x_sequence[-1], None, self.horizon)

        # Initialize gains
        k_sequence = np.zeros((self.horizon, self.control_dim))
        K_sequence = np.zeros((self.horizon, self.control_dim, self.state_dim))

        expected_cost_reduction = 0

        # Work backwards
        for t in range(self.horizon - 1, -1, -1):
            fx, fu, cost, cx, cu, cxx, cuu = self.compute_derivatives(x_sequence[t], u_sequence[t], t)

            # Q-function derivatives
            Qx = cx + fx.T @ Vx
            Qu = cu + fu.T @ Vx
            Qxx = cxx + fx.T @ Vxx @ fx
            Quu = cuu + fu.T @ Vxx @ fu
            Qux = fu.T @ Vxx @ fx

            # Regularization for numerical stability
            Quu_reg = Quu + self.reg * np.eye(self.control_dim)

            while not np.all(np.linalg.eigvals(Quu_reg) > 0):
                self.reg *= self.reg_factor
                if self.reg > self.reg_max:
                    print("Regularization too large, breaking")
                    break
                Quu_reg = Quu + self.reg * np.eye(self.control_dim)

            # Gains
            try:
                k = -np.linalg.solve(Quu_reg, Qu)
                K = -np.linalg.solve(Quu_reg, Qux)
            except np.linalg.LinAlgError:
                print("Matrix inversion failed")
                return None, None, False

            k_sequence[t] = k
            K_sequence[t] = K

            # Update value function derivatives for next iteration
            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

            Vxx = 0.5 * (Vxx + Vxx.T)

            expected_cost_reduction += 0.5 * k.T @ Quu @ k + k.T @ Qu

        return k_sequence, K_sequence, expected_cost_reduction

    def forward_pass(self, x0, x_sequence, u_sequence, k_sequence, K_sequence):
        alpha = self.alpha
        x_new = np.zeros((self.horizon + 1, self.state_dim))
        u_new = np.zeros((self.horizon, self.control_dim))

        x_new[0] = x0

        while alpha >= self.alpha_min:
            total_cost = 0

            # Control updates with line search
            for t in range(self.horizon):
                # State feedback control update
                delta_x = x_new[t] - x_sequence[t]
                u_new[t] = u_sequence[t] + alpha * k_sequence[t] + K_sequence[t] @ delta_x

                # Cost
                cost, _, _, _, _ = self.cost_func(x_new[t], u_new[t], t)
                total_cost += cost

                # Propagate dynamics
                x_new[t+1] = self.dynamics_func(x_new[t], u_new[t])

            # Terminal cost
            terminal_cost, _, _, _, _ = self.cost_func(x_new[-1], None, self.horizon)
            total_cost += terminal_cost

            # If cost improved, accept
            _, prev_cost = self.rollout(x0, u_sequence)
            if total_cost < prev_cost:
                return x_new, u_new, total_cost, True

            # Otherwise, reduce step size
            alpha *= self.alpha_decay

        # If no improvement, return failure
        return x_sequence, u_sequence, None, False

    def optimize(self, x0, initial_controls=None):
        if initial_controls is None:
            u_sequence = np.zeros((self.horizon, self.control_dim))
        else:
            u_sequence = initial_controls.copy()

        # Initial rollout
        x_sequence, total_cost = self.rollout(x0, u_sequence)

        # DDP iteration
        for iteration in range(self.max_iterations):
            self.reg = self.reg_min

            # Backward pass
            k_sequence, K_sequence, expected_reduction = self.backward_pass(x_sequence, u_sequence)
            if k_sequence is None:
                print("Backward pass failed")
                break

            # Check for convergence
            if abs(expected_reduction) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            # Forward pass with line search
            x_new, u_new, new_cost, improved = self.forward_pass(x0, x_sequence, u_sequence, k_sequence, K_sequence)

            if improved:
                x_sequence = x_new
                u_sequence = u_new

                if iteration % 10 == 0:
                    print(f"Iteration {iteration}, Cost: {new_cost}, Expected Reduction: {expected_reduction}")
            else:
                print(f"Line search failed at iteration {iteration}")
                break

        # Value function information at the first state for risk-sensitive EKF
        _, Vx, _, Vxx, _ = self.cost_func(x_sequence[0], u_sequence[0], 0)

        return x_sequence, u_sequence, K_sequence, Vx, Vxx
