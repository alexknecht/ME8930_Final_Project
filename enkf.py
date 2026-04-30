import numpy as np

class KoopmanEnKF:
    def __init__(self, K_baseline, n_ensemble=50, process_noise_std=1e-4, measurement_noise_std=1e-2):
        """
        Initializes the Ensemble Kalman Filter for Parameter Estimation.
        """
        self.K_shape = K_baseline.shape
        self.n_params = K_baseline.size  # E.g., 7x11 = 77 parameters
        self.N = n_ensemble
        
        # INITIALIZE THE PRIOR
        # Flatten the baseline K matrix to act as the mean of our prior distribution
        theta_mean = K_baseline.flatten()
        
        # Create an ensemble (cloud) of N different parameter vectors
        # Shape becomes (n_params, N) -> (77, 50)
        self.ensemble = np.tile(theta_mean, (self.N, 1)).T 
        
        # Add initial uncertainty (Gaussian noise) to create the distribution
        self.ensemble += np.random.normal(0, process_noise_std, self.ensemble.shape)
        
        # Define Noise Covariances
        # Q: Process Noise (How fast we think the vehicle dynamics change over time)
        self.Q = np.eye(self.n_params) * (process_noise_std**2)
        # R: Measurement Noise (How noisy the comma.ai sensors are)
        self.R = np.eye(1) * (measurement_noise_std**2) 

    def step(self, x_lifted_k, u_k, d_k, y_true_next):
        omega = np.concatenate((x_lifted_k, [u_k], d_k)) 
        
        process_noise = np.random.multivariate_normal(np.zeros(self.n_params), self.Q, self.N).T
        self.ensemble += process_noise
        
        # Reshape the (77, 50) ensemble into a 3D Tensor: (50 matrices, 7 rows, 11 cols)
        K_tensor = self.ensemble.T.reshape(self.N, 7, 11)
        
        # Numpy broadcasting automatically multiplies all 50 matrices by the omega vector in C!
        x_next_pred_all = K_tensor @ omega  # Shape becomes (50, 7)
        
        # Grab row 0 (the raw lateral acceleration) for all 50 predictions instantly
        Y_pred = x_next_pred_all[:, 0]
            
        Y_mean = np.mean(Y_pred)
        theta_mean = np.mean(self.ensemble, axis=1, keepdims=True)
        
        Y_dev = Y_pred - Y_mean
        theta_dev = self.ensemble - theta_mean
        
        P_yy = (Y_dev @ Y_dev.T) / (self.N - 1) + self.R[0,0]
        P_theta_y = (theta_dev @ Y_dev.T) / (self.N - 1)
        
        K_gain = P_theta_y / P_yy  
        
        y_perturbed = y_true_next + np.random.normal(0, np.sqrt(self.R[0,0]), self.N)

        innovations = y_perturbed - Y_pred # Shape: (50,)
        
        # np.outer instantly multiplies the (77,) gain by the (50,) innovations
        # to apply the exact shift to all 3,850 parameters simultaneously.
        self.ensemble += np.outer(K_gain, innovations)
            
        return np.mean(self.ensemble, axis=1).reshape(self.K_shape)
