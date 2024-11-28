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
        print(dt)
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