import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# NOMINAL ROCKET PARAMETERS
# ==========================================================
I_nom = 0.025        # kg·m^2
T_nom = 15.0         # N
L_nom = 0.12         # m

# ==========================================================
# CONTROL DESIGN TARGETS
# ==========================================================
zeta = 0.7
omega_n = 20.0       # rad/s

Kp = (I_nom * omega_n**2) / (T_nom * L_nom)
Kd = (2 * zeta * I_nom * omega_n) / (T_nom * L_nom)

print(f"Nominal gains: Kp={Kp:.2f}, Kd={Kd:.2f}")

# ==========================================================
# SIMULATION PARAMETERS
# ==========================================================
dt = 0.001
t_end = 0.75
time = np.arange(0, t_end, dt)

# ==========================================================
# ACTUATOR LIMITS (REALISM)
# ==========================================================
delta_max = np.deg2rad(5)      # gimbal angle limit
delta_rate_max = np.deg2rad(300)  # deg/s → rad/s

# ==========================================================
# SENSOR NOISE (IMU)
# ==========================================================
angle_noise_std = np.deg2rad(0.15)     # angle noise
rate_noise_std = np.deg2rad(0.8)       # gyro noise

# ==========================================================
# MONTE CARLO SETTINGS
# ==========================================================
N_runs = 100

I_sigma = 0.15
T_sigma = 0.10
L_sigma = 0.05

# ==========================================================
# STORAGE
# ==========================================================
theta_runs = []
psi_runs = []

# ==========================================================
# MONTE CARLO LOOP
# ==========================================================
for _ in range(N_runs):

    # Randomized parameters
    I = I_nom * (1 + np.random.randn() * I_sigma)
    T = T_nom * (1 + np.random.randn() * T_sigma)
    L = L_nom * (1 + np.random.randn() * L_sigma)

    # True states
    theta, theta_dot = np.deg2rad(5), 0.0
    psi, psi_dot = np.deg2rad(-4), 0.0

    # Actuator states
    delta_theta = 0.0
    delta_psi = 0.0

    theta_hist = []
    psi_hist = []

    for t in time:

        # ======================
        # SENSOR MEASUREMENTS
        # ======================
        theta_meas = theta + np.random.randn() * angle_noise_std
        theta_dot_meas = theta_dot + np.random.randn() * rate_noise_std

        psi_meas = psi + np.random.randn() * angle_noise_std
        psi_dot_meas = psi_dot + np.random.randn() * rate_noise_std

        # ======================
        # CONTROLLER (USES NOISY STATES)
        # ======================
        delta_theta_cmd = -Kp * theta_meas - Kd * theta_dot_meas
        delta_psi_cmd = -Kp * psi_meas - Kd * psi_dot_meas

        delta_theta_cmd = np.clip(delta_theta_cmd, -delta_max, delta_max)
        delta_psi_cmd = np.clip(delta_psi_cmd, -delta_max, delta_max)

        # ======================
        # ACTUATOR RATE LIMITS
        # ======================
        dtheta = delta_theta_cmd - delta_theta
        dpsi = delta_psi_cmd - delta_psi

        max_step = delta_rate_max * dt

        dtheta = np.clip(dtheta, -max_step, max_step)
        dpsi = np.clip(dpsi, -max_step, max_step)

        delta_theta += dtheta
        delta_psi += dpsi

        # ======================
        # DISTURBANCE TORQUES
        # ======================
        tau_theta = 0.0
        tau_psi = 0.0
        if 0.35 < t < 0.40:
            tau_theta = 0.02
            tau_psi = -0.015

        # ======================
        # DYNAMICS
        # ======================
        theta_ddot = (T * L * delta_theta + tau_theta) / I
        psi_ddot = (T * L * delta_psi + tau_psi) / I

        theta_dot += theta_ddot * dt
        theta += theta_dot * dt

        psi_dot += psi_ddot * dt
        psi += psi_dot * dt

        theta_hist.append(theta)
        psi_hist.append(psi)

    theta_runs.append(np.rad2deg(theta_hist))
    psi_runs.append(np.rad2deg(psi_hist))

theta_runs = np.array(theta_runs)
psi_runs = np.array(psi_runs)

# ==========================================================
# PLOTTING — MONTE CARLO ENVELOPES
# ==========================================================
def plot_envelope(data, title, ylabel):
    mean = np.mean(data, axis=0)
    low = np.percentile(data, 5, axis=0)
    high = np.percentile(data, 95, axis=0)

    plt.figure()
    plt.plot(time, mean, label="Mean Response")
    plt.fill_between(time, low, high, alpha=0.3, label="90% Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

plot_envelope(theta_runs,
              "Monte Carlo Pitch Response with Actuator & Sensor Realism",
              "Pitch Angle (deg)")

plot_envelope(psi_runs,
              "Monte Carlo Yaw Response with Actuator & Sensor Realism",
              "Yaw Angle (deg)")
