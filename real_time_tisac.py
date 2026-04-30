import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifter import FourierLifter
from enkf import KoopmanEnKF

#train baseline
print("Training offline baseline on 00000.csv...")
df_train = pd.read_csv('data/00000.csv').dropna()
N_tr = len(df_train) - 1

lifter = FourierLifter(num_frequencies=3)
X1_lifted = lifter.lift_trajectory(df_train['targetLateralAcceleration'].values[:N_tr].reshape(1, N_tr))
X2_lifted = lifter.lift_trajectory(df_train['targetLateralAcceleration'].values[1:].reshape(1, N_tr))
U_tr = df_train['steerCommand'].values[:N_tr].reshape(1, N_tr)
D_tr = np.vstack((df_train['vEgo'].values[:N_tr], df_train['aEgo'].values[:N_tr], df_train['roll'].values[:N_tr]))

Omega_tr = np.vstack((X1_lifted, U_tr, D_tr))
lambda_reg = 1e-4  
K_baseline = X2_lifted @ Omega_tr.T @ np.linalg.inv(Omega_tr @ Omega_tr.T + lambda_reg * np.eye(Omega_tr.shape[0]))

#initialize TISAC
print("Initializing EnKF with prior Koopman matrix...")
# We inject a bit of process noise to allow the matrix to adapt over time
enkf = KoopmanEnKF(K_baseline, n_ensemble=50, process_noise_std=1e-3, measurement_noise_std=1e-2)

#test on unseen segments
print("Running live simulation on unseen route 00001.csv...")
df_test = pd.read_csv('data/00001.csv').dropna()

lataccel_te = df_test['targetLateralAcceleration'].values
steer_te = -df_test['steerCommand'].values
v_ego_te = df_test['vEgo'].values
a_ego_te = df_test['aEgo'].values
road_lataccel_te = np.sin(df_test['roll'].values) * 9.81

N_te = len(df_test) - 1

# Arrays to store our 1-step-ahead predictions
static_predictions = np.zeros(N_te + 1)
adaptive_predictions = np.zeros(N_te + 1)

static_predictions[0] = lataccel_te[0]
adaptive_predictions[0] = lataccel_te[0]

# Two separate physics engines: one rigid, one adaptive
K_static = K_baseline.copy()
K_adaptive = K_baseline.copy()

#first attempt at live control
for k in range(N_te):
    # 1. Read sensors at current time step
    x_k = lataccel_te[k]
    u_k = steer_te[k]
    d_k = np.array([v_ego_te[k], a_ego_te[k], road_lataccel_te[k]])
    
    # Lift the current state
    x_lifted_k = lifter.lift_state(x_k)
    omega_k = np.concatenate((x_lifted_k, [u_k], d_k))
    
    # 2. Predict the FUTURE (k+1) using CURRENT matrices
    static_next_lifted = K_static @ omega_k
    static_predictions[k+1] = static_next_lifted[0]
    
    adaptive_next_lifted = K_adaptive @ omega_k
    adaptive_predictions[k+1] = adaptive_next_lifted[0]
    
    # In a real car, time moves forward here. We now observe the TRUE k+1 state.
    if k < N_te - 1:
        y_true_next = lataccel_te[k+1]
        
        # The static model does nothing and learns nothing.
        # The EnKF calculates the error and mathematically recalibrates K_adaptive!
        K_adaptive = enkf.step(x_lifted_k, u_k, d_k, y_true_next)


plt.figure(figsize=(14, 7))
# Removed the [:-1] slices so all arrays match the exact length of lataccel_te
plt.plot(lataccel_te, label='True lataccel (Ground Truth)', color='black', linewidth=3, alpha=0.5)
plt.plot(static_predictions, label='Static Fourier Model (Failing)', color='red', linestyle='dashed', linewidth=2, alpha=0.8)
plt.plot(adaptive_predictions, label='Adaptive TISAC EnKF (Healing)', color='blue', linewidth=2)

# Calculate MSE using the full arrays
mse_static = np.mean((lataccel_te - static_predictions)**2)
mse_adaptive = np.mean((lataccel_te - adaptive_predictions)**2)

plt.title(f'Real-Time TISAC Parameter Tracking\nStatic MSE: {mse_static:.6f} | Adaptive MSE: {mse_adaptive:.6f}')
plt.xlabel('Time Step (k)')
plt.ylabel('Lateral Acceleration (G)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
