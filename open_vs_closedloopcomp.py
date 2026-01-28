import numpy as np
import matplotlib.pyplot as plt

USE_CONTROL = True


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

results = {}


def run_sim(use_control):
    theta = np.deg2rad(5)
    theta_dot = 0.0
    
    theta_hist = []
    
    for t in time:
        if use_control:
            delta_cmd = -Kp * theta - Kd * theta_dot
        else:
            delta_cmd = 0.0

        delta = np.clip(delta_cmd, -delta_max, delta_max)
        theta_ddot = (T * L * delta) / I

        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        theta_hist.append(theta)
        
        

    return np.rad2deg(theta_hist)



theta_closed = run_sim(True)
theta_open = run_sim(False)

plt.figure()
plt.plot(time, theta_closed, label="Closed-loop (TVC)")
plt.plot(time, theta_open, '--', label="Open-loop")
plt.xlabel("Time (s)")
plt.ylabel("Pitch Angle (deg)")
plt.title("Open-loop vs Closed-loop Attitude Response")
plt.legend()
plt.grid()
plt.show()




delta_cmd = 0.0

