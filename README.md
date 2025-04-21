# Mobile-Robotics

# Risk-Sensitive Extended Kalman Filter (RS-EKF)

This repository contains an implementation of the Risk-Sensitive Extended Kalman Filter (RS-EKF) algorithm as described in the 2024 IEEE International Conference on Robotics and Automation (ICRA) paper by Jordana et al.

## Project Overview

The implementation includes:
- Extended Kalman Filter (EKF)
- Risk-Sensitive Extended Kalman Filter (RS-EKF)
- Differential Dynamic Programming (DDP) for Model Predictive Control (MPC)

The algorithms are tested on two platforms:
1. A unicycle model with friction uncertainties
2. A planar quadrotor with unknown load conditions

## Key Features

- **RS-EKF Algorithm**: A novel approach that integrates control objectives directly into the state estimation process
- **DDP-MPC Integration**: A complete implementation of Differential Dynamic Programming for Model Predictive Control
- **Comparative Analysis**: Evaluation of standard EKF vs. RS-EKF performance in uncertain environments

## Algorithm Architecture

The system follows a closed-loop architecture with four main components:
1. **Risk-Sensitive Extended Kalman Filter**: Processes sensor measurements and control commands to generate state estimates
2. **Model Predictive Controller (DDP)**: Computes optimal control sequences based on the risk-sensitive state estimates
3. **Robot Model**: Simulates the dynamics of the system (unicycle or quadrotor)
4. **Sensors**: Provide position and pose measurements

## Mathematical Formulation

### Risk-Sensitive EKF
The key innovation of RS-EKF is the risk-sensitive correction:

```
xˆRS = x̄ + (I - μPVxx)^(-1)(Δx̂ + μPvx)
```

Where:
- `xˆRS` is the risk-sensitive state estimate
- `x̄` is the predicted state estimate
- `μ` is the risk-sensitivity parameter
- `P` is the covariance matrix
- `Vxx` is the Hessian of the value function
- `vx` is the gradient of the value function
- `Δx̂` is the standard Kalman filter correction term

### Differential Dynamic Programming
The DDP algorithm computes optimal control sequences by iterating between backward and forward passes, calculating:
- Value function derivatives
- Feedback control gains
- Optimal control updates

## Test Cases

### 1. Unicycle with Friction Uncertainty
- State vector: position (x, y), orientation (θ), and friction coefficient (μf)
- Control inputs: linear velocity (v) and angular velocity (ω)
- Results show ~14.68% improvement in position error with RS-EKF

### 2. Planar Quadrotor with Unknown Load
- State vector: position (x, y), orientation (θ), velocities (vx, vy, ω), and mass (m)
- Control inputs: thrust forces from two rotors
- Results show ~5.7% reduction in position MSE and 3.6% improvement in trajectory cost

## Results

The implementation demonstrates that RS-EKF consistently outperforms standard EKF in uncertain conditions, particularly during sudden parameter changes:
- Faster convergence during initial parameter estimation
- More responsive behavior at transition points
- Quicker return to steady state compared to standard EKF

## Limitations and Future Work

- Sensitivity to parameter tuning, especially the risk-sensitivity parameter μ
- Computational complexity of calculating value function derivatives
- Potential extensions:
  - Adaptive risk parameters based on measured uncertainty
  - Integration with learning-based approaches for uncertainty modeling
  - Expansion to higher-dimensional systems with multiple uncertain parameters

## Dependencies

- Python 3.11.12
- NumPy
- SciPy
- Matplotlib (for visualization)

## References

[1] A. Jordana, A. Meduri, E. Arlaud, J. Carpentier, and L. Righetti, "Risk-Sensitive Extended Kalman Filter," in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 10450–10456.

## Author

Shalini Agrawal  
Northeastern University  
NUID: 002319438
