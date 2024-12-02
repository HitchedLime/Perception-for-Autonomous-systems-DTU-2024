import numpy as np

class IMMFilter3D:
    def __init__(self, initial_state):
        """
        Initialize IMM filter with Constant Velocity and Constant Acceleration models.
        
        Args:
            initial_state: [x, vx, y, vy, z, vz] initial state vector
        """
        # Model transition probability matrix
        self.model_trans_prob = np.array([
            [0.95, 0.05],  # Probability of staying in CV or switching to CA
            [0.05, 0.95]   # Probability of switching to CV or staying in CA
        ])
        
        # Initial model probabilities [CV, CA]
        self.model_prob = np.array([0.5, 0.5])
        
        # Initialize CV Model (state: [x, vx, y, vy, z, vz])
        self.cv_state = np.zeros(6)
        self.cv_state[:6] = initial_state
        self.cv_cov = np.eye(6) * 1000
        
        # Initialize CA Model (state: [x, vx, ax, y, vy, ay, z, vz, az])
        self.ca_state = np.zeros(9)
        self.ca_state[0] = initial_state[0]  # x
        self.ca_state[1] = initial_state[1]  # vx
        self.ca_state[3] = initial_state[2]  # y
        self.ca_state[4] = initial_state[3]  # vy
        self.ca_state[6] = initial_state[4]  # z
        self.ca_state[7] = initial_state[5]  # vz
        self.ca_cov = np.eye(9) * 1000
        
        # Measurement noise (position only)
        self.R = np.eye(3) * 0.1
        
        # Process noise parameters
        self.cv_q = 1.0  # CV model noise
        self.ca_q = 0.1  # CA model noise

    def predict(self, dt):
        """
        Predict step of IMM filter.
        
        Args:
            dt: Time step in seconds
        
        Returns:
            np.ndarray: Predicted position [x, y, z]
        """
        # Get model matrices
        F_cv, Q_cv = self._get_cv_matrices(dt)
        F_ca, Q_ca = self._get_ca_matrices(dt)
        
        # Mixing probabilities
        mixing_prob = self.model_trans_prob * self.model_prob.reshape(-1, 1)
        mixing_prob = mixing_prob / np.sum(mixing_prob, axis=1)[:, np.newaxis]
        
        # Mix states
        cv_mixed_state = self.cv_state.copy()
        cv_mixed_cov = self.cv_cov.copy()
        
        ca_mixed_state = self.ca_state.copy()
        ca_mixed_cov = self.ca_cov.copy()
        
        # Predict each model
        # CV Model
        self.cv_state = F_cv @ cv_mixed_state
        self.cv_cov = F_cv @ cv_mixed_cov @ F_cv.T + Q_cv
        
        # CA Model
        self.ca_state = F_ca @ ca_mixed_state
        self.ca_cov = F_ca @ ca_mixed_cov @ F_ca.T + Q_ca
        
        # Update model probabilities
        c = np.sum(self.model_prob)
        self.model_prob = self.model_prob / c if c > 0 else np.array([0.5, 0.5])
        
        # Combined state estimate (position only)
        cv_pos = np.array([self.cv_state[0], self.cv_state[2], self.cv_state[4]])
        ca_pos = np.array([self.ca_state[0], self.ca_state[3], self.ca_state[6]])
        
        # Ensure model probabilities are scalars for multiplication
        cv_prob = float(self.model_prob[0])
        ca_prob = float(self.model_prob[1])
        
        return cv_prob * cv_pos + ca_prob * ca_pos

    def update(self, measurement):
        """Update step of IMM filter."""
        z = measurement.reshape(3, 1)
        
        # CV Model update
        H_cv = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        # CA Model update
        H_ca = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        # Update CV Model
        S_cv = H_cv @ self.cv_cov @ H_cv.T + self.R
        K_cv = self.cv_cov @ H_cv.T @ np.linalg.inv(S_cv)
        self.cv_state = self.cv_state + K_cv @ (z - H_cv @ self.cv_state.reshape(-1, 1)).reshape(-1)
        self.cv_cov = self.cv_cov - K_cv @ H_cv @ self.cv_cov
        
        # Update CA Model
        S_ca = H_ca @ self.ca_cov @ H_ca.T + self.R
        K_ca = self.ca_cov @ H_ca.T @ np.linalg.inv(S_ca)
        self.ca_state = self.ca_state + K_ca @ (z - H_ca @ self.ca_state.reshape(-1, 1)).reshape(-1)
        self.ca_cov = self.ca_cov - K_ca @ H_ca @ self.ca_cov
        
        # Update model probabilities using likelihood
        cv_likelihood = self._gaussian_likelihood(z, H_cv @ self.cv_state.reshape(-1, 1), S_cv)
        ca_likelihood = self._gaussian_likelihood(z, H_ca @ self.ca_state.reshape(-1, 1), S_ca)
        
        c = np.sum(np.array([cv_likelihood, ca_likelihood]) * self.model_prob)
        if c > 0:
            self.model_prob = np.array([cv_likelihood, ca_likelihood]) * self.model_prob / c
        else:
            self.model_prob = np.array([0.5, 0.5])
        
        # Combined state estimate (position only)
        cv_pos = np.array([self.cv_state[0], self.cv_state[2], self.cv_state[4]])
        ca_pos = np.array([self.ca_state[0], self.ca_state[3], self.ca_state[6]])
        
        # Ensure model probabilities are scalars for multiplication
        cv_prob = float(self.model_prob[0])
        ca_prob = float(self.model_prob[1])
        
        return cv_prob * cv_pos + ca_prob * ca_pos

    def _get_cv_matrices(self, dt):
        """Get state transition and process noise matrices for CV model."""
        # State transition matrix
        F = np.array([
            [1, dt, 0, 0,  0, 0],
            [0, 1,  0, 0,  0, 0],
            [0, 0,  1, dt, 0, 0],
            [0, 0,  0, 1,  0, 0],
            [0, 0,  0, 0,  1, dt],
            [0, 0,  0, 0,  0, 1]
        ])
        
        # Process noise matrix
        q = self.cv_q
        Q = np.array([
            [dt**4/4, dt**3/2, 0, 0, 0, 0],
            [dt**3/2, dt**2,   0, 0, 0, 0],
            [0, 0, dt**4/4, dt**3/2, 0, 0],
            [0, 0, dt**3/2, dt**2,   0, 0],
            [0, 0, 0, 0, dt**4/4, dt**3/2],
            [0, 0, 0, 0, dt**3/2, dt**2]
        ]) * q
        
        return F, Q

    def _get_ca_matrices(self, dt):
        """Get state transition and process noise matrices for CA model."""
        # State transition matrix
        F = np.array([
            [1, dt, dt**2/2, 0, 0, 0, 0, 0, 0],
            [0, 1,  dt,      0, 0, 0, 0, 0, 0],
            [0, 0,  1,       0, 0, 0, 0, 0, 0],
            [0, 0,  0,       1, dt, dt**2/2, 0, 0, 0],
            [0, 0,  0,       0, 1, dt, 0, 0, 0],
            [0, 0,  0,       0, 0, 1, 0, 0, 0],
            [0, 0,  0,       0, 0, 0, 1, dt, dt**2/2],
            [0, 0,  0,       0, 0, 0, 0, 1, dt],
            [0, 0,  0,       0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise matrix
        q = self.ca_q
        Q = np.zeros((9, 9))
        for i in range(3):  # For each coordinate (x, y, z)
            idx = i * 3
            Q[idx:idx+3, idx:idx+3] = np.array([
                [dt**4/4, dt**3/2, dt**2/2],
                [dt**3/2, dt**2,   dt],
                [dt**2/2, dt,      1]
            ]) * q
            
        return F, Q

    def _gaussian_likelihood(self, z, z_pred, S):
        """Calculate Gaussian likelihood of measurement."""
        dim = z.shape[0]
        diff = (z - z_pred)
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        likelihood = 1.0 / np.sqrt((2 * np.pi) ** dim * det_S) * \
                    np.exp(-0.5 * diff.T @ inv_S @ diff)
        return likelihood[0, 0]