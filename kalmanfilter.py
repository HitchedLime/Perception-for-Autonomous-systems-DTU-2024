import numpy as np

class MeasurementCovarianceEstimator:
    def __init__(self, window_size=10):
        """
        Collect centroid measurements across multiple frames to estimate covariance
        
        Args:
        window_size (int): Number of frames to collect for covariance estimation
        """
        self.window_size = window_size
        self.measurements = {
            0: [],  # car
            1: [],  # person
            2: []   # bike (adjust based on your class labels)
        }
    
    def add_measurement(self, centroid, class_label):
        """
        Add a new centroid measurement for a specific class
        
        Args:
        centroid (np.ndarray): 3D centroid coordinates
        class_label (int): Class of the object
        """
        # Ensure the class label exists in our measurements dict
        if class_label not in self.measurements:
            self.measurements[class_label] = []
        
        # Add measurement
        self.measurements[class_label].append(centroid)
        
        # Trim to window size
        if len(self.measurements[class_label]) > self.window_size:
            self.measurements[class_label] = self.measurements[class_label][-self.window_size:]
    
    def get_measurement_covariance(self, class_label):
        """
        Calculate measurement covariance for a specific class
        
        Args:
        class_label (int): Class to calculate covariance for
        
        Returns:
        np.ndarray: 3x3 covariance matrix, or identity matrix if insufficient data
        """
        # Get measurements for this class
        class_measurements = self.measurements.get(class_label, [])
        
        # Check if we have enough measurements
        if len(class_measurements) < 2:
            return np.eye(3) * 10  # Default high uncertainty
        
        # Convert to numpy array and calculate covariance
        measurements_array = np.array(class_measurements)
        return np.cov(measurements_array.T)

class KalmanFilter3D_with_cov_est:
    def __init__(self, initial_state=None, class_label=None, cov_estimator=None):
        """
        Initialize 3D Kalman Filter with constant velocity model
        
        State vector: [x, vx, y, vy, z, vz]
        """
        # Transition matrix (constant velocity model)
        self.F = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Observation matrix (map state to measurement)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        # Measurement noise covariance
        self.R = np.eye(3) * 10  # Adjust based on sensor noise
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Identity matrix
        self.I = np.eye(6)
        
        # Initial state and covariance
        if initial_state is None:
            self.x = np.zeros((6, 1))
        else:
            self.x = initial_state.reshape((6, 1))
        
        self.P = np.eye(6) * 1000  # High initial uncertainty

        self.class_label = class_label
        self.cov_estimator = cov_estimator
        
        # Dynamic measurement noise
        if cov_estimator and class_label is not None:
            self.R = cov_estimator.get_measurement_covariance(class_label)
        else:
            # Fallback to default
            self.R = np.eye(3) * 10
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].flatten()
    
    def update(self, z):
        """
        Update state based on measurement
        
        Args:
            z (np.ndarray): Measurement vector [x, y, z]
        """
        z = z.reshape((3, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        
        return self.x[:3].flatten()
    

class KalmanFilter3D:
    def __init__(self, initial_state=None):
        """
        Initialize 3D Kalman Filter with constant velocity model
        State vector: [x, vx, y, vy, z, vz]
        
        Args:
            initial_state: Initial state vector [x, vx, y, vy, z, vz]
        """
        # Initialize state
        if initial_state is None:
            self.x = np.zeros((6, 1))
        else:
            # Ensure proper shape and handle velocity initialization
            pos = initial_state[:3]
            vel = initial_state[3:] if len(initial_state) > 3 else np.zeros(3)
            self.x = np.array([
                pos[0], vel[0],  # x, vx
                pos[1], vel[1],  # y, vy
                pos[2], vel[2]   # z, vz
            ]).reshape((6, 1))
        
        # Initialize covariance with higher uncertainty for velocity
        self.P = np.diag([
            10, 100,  # x, vx uncertainty
            10, 100,  # y, vy uncertainty
            10, 100   # z, vz uncertainty
        ])
        
        # Measurement noise - adjust based on your sensor characteristics
        self.R = np.diag([1e-2, 1e-2, 1e-2])  # Position measurement noise
        
        # Observation matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        self.I = np.eye(6)
        
        # Previous timestamp for dt calculation
        self.last_update_time = None
        
    def predict(self, dt):
        """
        Predict next state using actual time difference
        
        Args:
            dt: Time difference in seconds
        """
        # Limit dt to reasonable values to prevent instability
        dt = min(max(dt, 0.01), 1.0)
        # print(dt)
        # State transition matrix
        self.F = np.array([
            [1, dt, 0,  0, 0,  0],
            [0,  1, 0,  0, 0,  0],
            [0,  0, 1, dt, 0,  0],
            [0,  0, 0,  1, 0,  0],
            [0,  0, 0,  0, 1, dt],
            [0,  0, 0,  0, 0,  1]
        ])
        
        # Process noise grows with dt
        q = 0.1  # Process noise parameter
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        self.Q = q * np.array([
            [dt4/4, dt3/2,     0,     0,     0,     0],
            [dt3/2,   dt2,     0,     0,     0,     0],
            [    0,     0, dt4/4, dt3/2,     0,     0],
            [    0,     0, dt3/2,   dt2,     0,     0],
            [    0,     0,     0,     0, dt4/4, dt3/2],
            [    0,     0,     0,     0, dt3/2,   dt2]
        ])
        
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:5:2].flatten()  # Return position only [x, y, z]
    
    def update(self, measurement):
        """
        Update state based on measurement
        
        Args:
            measurement: Position measurement [x, y, z]
        """
        z = measurement.reshape((3, 1))
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (self.I - K @ self.H) @ self.P
        
        return self.x[:5:2].flatten()  # Return position only [x, y, z]
    
import numpy as np

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