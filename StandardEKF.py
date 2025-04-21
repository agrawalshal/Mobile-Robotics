import numpy as np

class ExtendedKalmanFilter:
  def __init__(
      self,
      state_dim=3, #[x,y,theta]
      measurement_dim=2, #[x,y]
      dt=0.1
  ):

    self.state_dim = state_dim
    self.measurement_dim = measurement_dim
    self.dt = dt

    self.state_mean = np.zeros(state_dim)
    self.state_covariance = np.eye(state_dim)
    self.R = np.diag([0.01, 0.01, 0.001])  # Process noise covariance
    self.Q = np.diag([0.01, 0.01]) #Measurement Covariance

  def motion_model(self, state, control): #Unicycle model

    process_noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.R)
    x, y, theta = state
    v, w = control

    new_x = x + v * np.cos(theta) * self.dt + process_noise[0]
    new_y = y + v * np.sin(theta) * self.dt + process_noise[1]
    new_theta = theta + w * self.dt + process_noise[2]

    state = np.array([new_x, new_y, new_theta])

    return state

  def measurement_model(self, state):

    measurement_noise = np.random.multivariate_normal(np.zeros(self.measurement_dim), self.Q)
    z = state[:self.measurement_dim]  + measurement_noise

    return z

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

  def update(self, z, vx=None, Vxx=None):

    H = self.jacobian_measurement_model(self.state_mean)
    K_t = self.state_covariance @ H.T @ np.linalg.inv(H @ self.state_covariance @ H.T + self.Q)
    self.state_mean = self.state_mean + K_t @ (z - self.measurement_model(self.state_mean))
    self.state_covariance = (np.eye(self.state_dim) - K_t @ H) @ self.state_covariance

    return self.state_mean, self.state_covariance
