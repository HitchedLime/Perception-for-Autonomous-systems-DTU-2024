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
        Initialize 3D Kalman Filter with constant velocity model and variable dt
        State vector: [x, vx, y, vy, z, vz]
        """
        # We'll update F in predict() with the actual dt
        self.F = None
        
        # Other matrices remain the same
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        self.R = np.eye(3) * 10  
        self.Q = np.eye(6) * 0.1
        self.I = np.eye(6)
        
        if initial_state is None:
            self.x = np.zeros((6, 1))
        else:
            self.x = initial_state.reshape((6, 1))
        
        self.P = np.eye(6) * 1000

    def predict(self, dt):
        """
        Predict next state using actual time difference
        
        Args:
            dt: Time difference between current and previous frame
        """
        # Update F matrix with actual dt
        self.F = np.array([
            [1, dt, 0,  0,  0,  0 ],
            [0,  1, 0,  0,  0,  0 ],
            [0,  0, 1, dt,  0,  0 ],
            [0,  0, 0,  1,  0,  0 ],
            [0,  0, 0,  0,  1, dt ],
            [0,  0, 0,  0,  0,  1 ]
        ])

        # Process noise Q should also be scaled with dt
        # This is a simple model for process noise that grows with time
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        # Position uncertainty grows with dt^4/4, velocity with dt^2/2
        q = 0.1  # process noise parameter
        self.Q = q * np.array([
            [dt4/4, dt3/2,     0,     0,     0,     0],
            [dt3/2,   dt2,     0,     0,     0,     0],
            [    0,     0, dt4/4, dt3/2,     0,     0],
            [    0,     0, dt3/2,   dt2,     0,     0],
            [    0,     0,     0,     0, dt4/4, dt3/2],
            [    0,     0,     0,     0, dt3/2,   dt2]
        ])

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