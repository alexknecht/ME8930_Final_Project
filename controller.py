import numpy as np
from lifter import FourierLifter
from enkf import KoopmanEnKF
from smpc import StochasticMPC

class Controller:
    def __init__(self):
        """
        Initializes the Real-Time TISAC Controller.
        """
        #Load the offline prior (The Fourier-Lifted DMDc Matrix)
        try:
            self.K_baseline = np.load('K_baseline.npy')
        except FileNotFoundError:
            raise FileNotFoundError("Please run train_fourier_koopman.py first to generate K_baseline.npy")

        #Initialize the TISAC Architecture Components
        self.lifter = FourierLifter(num_frequencies=3)
        self.enkf = KoopmanEnKF(self.K_baseline, n_ensemble=50, process_noise_std=1e-3, measurement_noise_std=1e-1)
        
        #set the horizon to 10 steps (1 second into the future at 10Hz)
        self.smpc = StochasticMPC()

        #Memory for the Recursive Filter
        # The EnKF needs to remember what it did *last* time to learn from its mistakes
        self.x_lifted_prev = None
        self.u_prev = 0.0
        self.d_prev = None

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        The main control loop called by the tinyphysics simulator at every time step.
        """
        # Extract current environmental disturbances from the simulator state
        v_ego = state.v_ego
        a_ego = state.a_ego
        road_roll = state.roll_lataccel
        d_curr = np.array([v_ego, a_ego, road_roll])

        # Lift the current lateral acceleration into the Fourier space
        x_lifted_curr = self.lifter.lift_state(current_lataccel)


        # We can only update if we have a past state to compare against current reality
        if self.x_lifted_prev is not None:
            # The EnKF calculates the error between its past prediction and current_lataccel
            # It then mathematically shifts the 50 Koopman matrices to fix the error!
            _ = self.enkf.step(
                x_lifted_k=self.x_lifted_prev,
                u_k=self.u_prev,
                d_k=self.d_prev,
                y_true_next=current_lataccel
            )

        # Extract the current, fully updated ensemble of matrices to feed the MPC
        K_ensemble = self.enkf.ensemble

        #MPC forecasting
        available_steps = len(future_plan.lataccel)
        
        future_targets = []
        future_disturbances = []
        
        for i in range(self.smpc.H):
            if available_steps > 0:
                # Safely pad the arrays if we are near the end
                idx = min(i, available_steps - 1)
                future_targets.append(future_plan.lataccel[idx])
                future_disturbances.append([
                    future_plan.v_ego[idx], 
                    future_plan.a_ego[idx], 
                    future_plan.roll_lataccel[idx]
                ])
            else:
                # Absolute end of simulation: future_plan is completely empty.
                # Hold the current target and physical state to finish the last milliseconds.
                future_targets.append(target_lataccel)
                future_disturbances.append([state.v_ego, state.a_ego, state.roll_lataccel])

        # Solve for the optimal, risk-aware steering angle
        optimal_steer = self.smpc.optimize_steering(
            x_lifted_curr=x_lifted_curr,
            u_prev=self.u_prev,
            target_lataccels=future_targets,
            disturbances=future_disturbances,
            K_ensemble=K_ensemble
        )


        # Save current states to act as the "past" during the next loop
        self.x_lifted_prev = x_lifted_curr
        self.u_prev = optimal_steer
        self.d_prev = d_curr

        return optimal_steer
