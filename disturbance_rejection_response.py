import numpy as np
import matplotlib.pyplot as plt

# =========================
# Physical Rocket Parameters
# =========================
I = 0.025        # Moment of inertia (kg·m^2)
T = 15.0         # Thrust (N)
L = 0.12         # CG to nozzle distance (m)

# =========================
# Control Design Parameters
# =========================
zeta = 0.7
omega_n = 20.0   # rad/s

Kp = (I * omega_n**2) / (T * L)
Kd = (2 * zeta * I * omega_n) / (T * L)

print(f"Kp = {Kp:.2f}, Kd = {Kd:.2f}")

# =========================
# Simulation Parameters
# =========================
dt = 0.001
t_end = 5.0
time = np.arange(0, t_end, dt)

delta_max = np.deg2rad(5)  # ±5° gimbal limit

# =========================
# Initial Conditions
# =========================
theta = np.deg2rad(8)   # 5° initial pitch error
theta_dot = 0.0

# =========================
# Data Storage
# =========================
theta_hist = []
theta_dot_hist = []
delta_hist = []

# =========================
# Simulation Loop
# =========================
for t in time:
    # PD Controller
    delta_cmd = -Kp * theta - Kd * theta_dot
    delta = np.clip(delta_cmd, -delta_max, delta_max)

    # Disturbance Torque (wind gust / rail departure)
    tau_dist = 0.0
    if 0.40 < t < 0.45:
        tau_dist = 0.02  # Nm impulse disturbance

    # Rotational Dynamics
    theta_ddot = (T * L * delta + tau_dist) / I

    # Integrate (Euler)
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # Store history
    theta_hist.append(theta)
    theta_dot_hist.append(theta_dot)
    delta_hist.append(delta)

# =========================
# Plot: Disturbance Rejection
# =========================
plt.figure()
plt.plot(time, np.rad2deg(theta_hist))
plt.xlabel("Time (s)")
plt.ylabel("Pitch Angle (deg)")
plt.title("Disturbance Rejection Response Under TVC Control")
plt.grid()
plt.show()
