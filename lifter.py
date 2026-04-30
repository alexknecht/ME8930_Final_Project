import numpy as np

class FourierLifter:
    def __init__(self, num_frequencies=5, base_freq=np.pi):
        """
        Initializes the Fourier Lifter.
        num_frequencies: How many sine/cosine pairs to generate.
        base_freq: The fundamental frequency to scale by.
        """
        self.num_frequencies = num_frequencies
        
        # Generates frequencies in powers of 2 (1, 2, 4, 8, 16...) 
        # This allows the Koopman operator to capture both broad curves and micro-vibrations.
        self.frequencies = (2.0 ** np.arange(num_frequencies)) * base_freq

    def lift_state(self, x):
        """
        Lifts a single state value (or array of values at one time step).
        Example: If x is [lataccel], and num_frequencies is 2, the output is:
        [lataccel, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x)]
        """
        # Always keep the raw, linear state as the foundation!
        lifted_state = [x]
        
        for freq in self.frequencies:
            lifted_state.append(np.sin(freq * x))
            lifted_state.append(np.cos(freq * x))
            
        return np.array(lifted_state).flatten()

    def lift_trajectory(self, X_matrix):
        """
        Vectorized lifting for the entire offline DMDc training matrix.
        X_matrix shape: (num_state_vars, num_timesteps)
        """
        lifted_rows = [X_matrix]
        
        for freq in self.frequencies:
            lifted_rows.append(np.sin(freq * X_matrix))
            lifted_rows.append(np.cos(freq * X_matrix))
            
        # Stacks everything vertically into a massive snapshot matrix
        return np.vstack(lifted_rows)

# --- Quick Sanity Check Test ---
if __name__ == "__main__":
    lifter = FourierLifter(num_frequencies=3)
    
    # Imagine our current lateral acceleration is 0.5 Gs
    raw_lataccel = np.array([0.5])
    
    lifted = lifter.lift_state(raw_lataccel)
    print(f"Raw State Dimension: {len(raw_lataccel)}")
    print(f"Lifted State Dimension: {len(lifted)}")
    print(f"Lifted Vector: \n{np.round(lifted, 3)}")