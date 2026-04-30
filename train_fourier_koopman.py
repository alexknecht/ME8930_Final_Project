import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifter import FourierLifter # Importing your new class!

# ==========================================
# 1. LOAD & PREP DATA (00000.csv)
# ==========================================
print("Loading data and initializing Fourier Lifter...")
df = pd.read_csv('data/00000.csv').dropna()

lataccel = df['targetLateralAcceleration'].values
steer = -df['steerCommand'].values
v_ego = df['vEgo'].values
a_ego = df['aEgo'].values
road_lataccel = np.sin(df['roll'].values) * 9.81

N = len(df) - 1

# ==========================================
# 2. THE LIFTING PHASE (The Magic Happens Here)
# ==========================================
# We use 3 frequencies: base, 2x, and 4x. 
# This turns our 1D state into a 7D state (1 raw + 3 sines + 3 cosines)
lifter = FourierLifter(num_frequencies=3)

# Extract and reshape raw states to (1, N)
X1_raw = lataccel[:N].reshape(1, N)
X2_raw = lataccel[1:].reshape(1, N)

# Lift the states! 
X1_lifted = lifter.lift_trajectory(X1_raw) # Shape becomes (7, N)
X2_lifted = lifter.lift_trajectory(X2_raw) # Shape becomes (7, N)

# Inputs and Disturbances remain un-lifted
U = steer[:N].reshape(1, N)
D = np.vstack((v_ego[:N], a_ego[:N], road_lataccel[:N]))

# Omega now contains the 7D state, 1D input, and 3D disturbance
Omega = np.vstack((X1_lifted, U, D)) # Shape: (11, N)

# ==========================================
# 3. SOLVE FOR THE HIGH-DIMENSIONAL KOOPMAN OPERATOR
# ==========================================
lambda_reg = 1e-4  
I = np.eye(Omega.shape[0]) 

# K will now be a 7x11 matrix!
K = X2_lifted @ Omega.T @ np.linalg.inv(Omega @ Omega.T + lambda_reg * I)

# Unpack the block matrices
# K_x describes how the 7 Fourier states interact with each other
K_x = K[:, 0:7]   # Shape: (7, 7)
K_u = K[:, 7:8]   # Shape: (7, 1)
K_d = K[:, 8:]    # Shape: (7, 3)

# Save the baseline matrix so our live controller can load it!
np.save('K_baseline.npy', K)

print(f"Koopman Matrix extracted successfully. Shape: {K.shape}")

# ==========================================
# 4. VALIDATION (Open-Loop Rollout)
# ==========================================
print("Running open-loop validation in the lifted space...")

# We must track the entire 7D state during the simulation
predicted_lifted_states = np.zeros((7, N))

# Give it the true first state, lifted
predicted_lifted_states[:, 0] = lifter.lift_state(lataccel[0])

for k in range(N - 1):
    x_k = predicted_lifted_states[:, k] # 7D vector
    u_k = steer[k]
    d_k = np.array([v_ego[k], a_ego[k], road_lataccel[k]])
    
    # The math remains elegantly linear, even though the physics are not!
    x_next = (K_x @ x_k) + (K_u @ [u_k]) + (K_d @ d_k)
    predicted_lifted_states[:, k+1] = x_next

# To get our actual lateral acceleration prediction, we just grab row 0 
# (Since row 0 of our lifted state is always the raw linear state)
predicted_lataccel = predicted_lifted_states[0, :]

# ==========================================
# 5. PLOT RESULTS
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(lataccel[:N], label='Actual lataccel (Ground Truth)', color='orange', linewidth=2, alpha=0.7)
plt.plot(predicted_lataccel, label='Fourier-Lifted DMDc Prediction', color='blue', linestyle='dashed', linewidth=2)

mse = np.mean((lataccel[:N] - predicted_lataccel)**2)
plt.title(f'Phase 1: Fourier-Lifted Koopman Baseline\nMean Squared Error: {mse:.6f}')
plt.xlabel('Time Step (k)')
plt.ylabel('Lateral Acceleration')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()