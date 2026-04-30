import numpy as np
from scipy.optimize import minimize

class StochasticMPC:
    #def __init__(self, horizon=5, w_err=45.0, w_jerk=10.0, w_steer=0.02, w_uncert=0.001):
    def __init__(self, horizon=5, w_err=45.0, w_jerk=10.0, w_steer=0.02, w_uncert=0.001):
        self.H = horizon
        self.w_err = w_err
        self.w_jerk = w_jerk
        self.w_steer = w_steer       
        self.w_uncert = w_uncert
        self._prev_solution = None
        
        self.steer_bounds = [(-2.0, 2.0) for _ in range(self.H)]
        """
        Initializes the Koopman-based Stochastic MPC.
        horizon: How many time steps into the future to predict.
        w_err: Penalty for missing the target path.
        w_jerk: Penalty for aggressive steering (passenger comfort).
        w_uncert: Penalty for steering into unknown physics (The TISAC novelty).
        """

    def optimize_steering(self, x_lifted_curr, u_prev, target_lataccels, disturbances, K_ensemble):
        N_ensemble = K_ensemble.shape[1]
        
        # Reshape the (77, 50) ensemble into a 3D Tensor: (50 matrices, 7 rows, 11 cols)
        K_tensor = K_ensemble.T.reshape(N_ensemble, 7, 11)

        def cost_function(U_guess):
            total_cost = 0.0
            
            # Copy the current state for all 50 universes
            # Shape becomes (50, 7)
            x_batch = np.tile(x_lifted_curr, (N_ensemble, 1))
            
            # Step forward through the horizon
            for k in range(self.H):
                # 1. Build the Omega vector for all 50 universes simultaneously
                u_batch = np.full((N_ensemble, 1), U_guess[k])                # Shape: (50, 1)
                d_batch = np.tile(disturbances[k], (N_ensemble, 1))           # Shape: (50, 3)
                
                # Concatenate along the columns to get the full state-input-disturbance vector
                omega_batch = np.concatenate([x_batch, u_batch, d_batch], axis=1) # Shape: (50, 11)
                
                # 2. Vectorized Matrix Multiplication
                # We add a dummy dimension to omega_batch to do batched matmul:
                # (50, 7, 11) @ (50, 11, 1) -> (50, 7, 1), then squeeze back to (50, 7)
                x_batch = (K_tensor @ omega_batch[..., None]).squeeze(-1)
                
                # 3. Extract Lateral Acceleration (Row 0) for all 50 universes
                y_preds = x_batch[:, 0]
                
                # 4. Calculate the Statistical Cost
                y_mean = np.mean(y_preds)
                y_var = np.var(y_preds)
                
                if k == 0:
                    jerk = U_guess[k] - u_prev
                else:
                    jerk = U_guess[k] - U_guess[k-1]
                
                step_cost = (self.w_err * (y_mean - target_lataccels[k])**2) + \
                            (self.w_jerk * (jerk)**2) + \
                            (self.w_steer * (U_guess[k])**2) + \
                            (self.w_uncert * y_var)
                            
                total_cost += step_cost
                
            return total_cost

        # Warm start the optimizer
        if self._prev_solution is not None:
            U0 = np.roll(self._prev_solution, -1)
            U0[-1] = u_prev  
        else:
            U0 = np.full(self.H, u_prev)

        # SLSQP Optimizer
        result = minimize(cost_function, U0, bounds=self.steer_bounds,
                        method='SLSQP', options={'ftol': 1e-3, 'maxiter': 20})

        self._prev_solution = result.x
        
        return result.x[0]
