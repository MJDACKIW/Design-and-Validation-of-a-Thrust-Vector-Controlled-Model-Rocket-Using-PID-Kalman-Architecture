import numpy as np
import matplotlib.pyplot as plt


# Physical parameters
I = 0.025        # kg·m² (example)
T = 15.0         # N
L = 0.12         # m

# Control design parameters
zeta = 0.7
omega_n = 20.0   # rad/s

# Derived PD gains
Kp = (I * omega_n**2) / (T * L)
Kd = (2 * zeta * I * omega_n) / (T * L)

print(f"Kp = {Kp:.2f}, Kd = {Kd:.2f}")

dt = 0.001
t_end = 2.0
time = np.arange(0, t_end, dt)

delta_max = np.deg2rad(5)   # ±5 degrees


theta = np.deg2rad(5)   # 5° initial error
theta_dot = 0.0

theta_hist = []
theta_dot_hist = []
delta_hist = []

for t in time:
    # PD Controller
    delta_cmd = -Kp * theta - Kd * theta_dot
    
    # Saturate gimbal
    delta = np.clip(delta_cmd, -delta_max, delta_max)
    
    # External disturbance (optional gust)
    tau_dist = 0.0
    if 0.3 < t < 0.35:
        tau_dist = 0.02  # Nm impulse disturbance
    
    # Dynamics
    theta_ddot = (T * L * delta + tau_dist) / I
    
    # Integrate
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt
    
    # Store
    theta_hist.append(theta)
    theta_dot_hist.append(theta_dot)
    delta_hist.append(delta)

plt.figure()
plt.plot(time, np.rad2deg(theta_dot_hist))
plt.xlabel("Time (s)")
plt.ylabel("Pitch Rate (deg/s)")
plt.title("Angular Rate Response Under PD Control")
plt.grid()
plt.show()


delta_cmd = 0.0

