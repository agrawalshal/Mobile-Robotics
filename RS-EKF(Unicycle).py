import numpy as np

class RiskSensitiveEKF:
  def __init__(
      self,
      state_dim=3, #[x,y,theta]
      measurement_dim=2, #[x,y]
      dt=0.1,
      mu=None, #risk sensitivity parameter
      vx=None,
      Vxx=None
  ):
    self.state_dim = state_dim
    self.measurement_dim = measurement_dim
    self.dt = dt

    self.state_mean = np.zeros(state_dim)
    self.state_covariance = np.eye(state_dim)

    self.R = np.diag([0.01, 0.01, 0.001])
    self.Q = np.diag([0.01, 0.01])

    if mu is None:
        self.mu = 0.0
    else:
        self.mu = mu

    # Initialize vx as a vector if provided, otherwise zero vector
    if vx is None:
        self.vx = np.zeros(state_dim)
    elif isinstance(vx, (int, float)):
        self.vx = np.ones(state_dim) * vx
    else:
        self.vx = vx

    # Initialize Vxx as a matrix if provided, otherwise zero matrix
    if Vxx is None:
        self.Vxx = np.zeros((state_dim, state_dim))
    elif isinstance(Vxx, (int, float)):
        self.Vxx = np.eye(state_dim) * Vxx
    else:
        self.Vxx = Vxx

  def motion_model(self, state, control):
    x, y, theta = state
    v, w = control

    new_x = x + v * np.cos(theta) * self.dt
    new_y = y + v * np.sin(theta) * self.dt
    new_theta = theta + w * self.dt

    return np.array([new_x, new_y, new_theta])

  def measurement_model(self, state):
    return state[:self.measurement_dim]

  def jacobian_motion_model(self, state, control):
    x, y, theta = state
    v, w = control

    G = np.eye(self.state_dim)
    G[0, 2] = -v * np.sin(theta) * self.dt
    G[1, 2] = v * np.cos(theta) * self.dt

    return G

  def jacobian_measurement_model(self, state):
    H = np.zeros((self.measurement_dim, self.state_dim))
    H[0, 0] = 1
    H[1, 1] = 1
    return H

  def predict(self, control):
    G = self.jacobian_motion_model(self.state_mean, control)

    self.state_mean = self.motion_model(self.state_mean, control)

    self.state_covariance = G @ self.state_covariance @ G.T + self.R

    return self.state_mean, self.state_covariance

  def update(self, z):
    H = self.jacobian_measurement_model(self.state_mean)

    expected_z = self.measurement_model(self.state_mean)

    innovation = z - expected_z

    S = H @ self.state_covariance @ H.T + self.Q

    K_t = self.state_covariance @ H.T @ np.linalg.inv(S)

    delta_x = K_t @ innovation

    self.state_covariance = (np.eye(self.state_dim) - K_t @ H) @ self.state_covariance

    # Standard EKF update if no risk sensitivity parameters are provided
    if np.all(self.vx == 0) and np.all(self.Vxx == 0):
        self.state_mean = self.state_mean + delta_x
    else:
        # Risk-sensitive update
        P_t = self.state_covariance
        I = np.eye(self.state_dim)
        temp_matrix = I - self.mu * P_t @ self.Vxx

        reg = 1e-6
        while not np.all(np.linalg.eigvals(temp_matrix) > 0):
            temp_matrix = I - self.mu * P_t @ (self.Vxx - reg * np.eye(self.state_dim))
            reg *= 10

        risk_shift = np.linalg.solve(temp_matrix, delta_x + self.mu * P_t @ self.vx)

        self.state_mean = self.state_mean + risk_shift

    return self.state_mean, self.state_covariance
