import numpy as np
from scipy.optimize import linear_sum_assignment
from kalmanfilter import *

class Detection:
    def __init__(self, centroid, class_label):
        self.centroid = centroid  # A NumPy array of shape (3,)
        self.class_label = class_label  # An integer or string representing the class

class TrackedObject:
    def __init__(self, detection: Detection, object_id, cov_estimator = None):
        self.id = object_id
        self.class_label = detection.class_label
        self.centroid = detection.centroid
        
        # Initialize Kalman filter with current centroid
        initial_state = np.array([
            detection.centroid[0], 0,  # x, vx
            detection.centroid[1], 0,  # y, vy
            detection.centroid[2], 0   # z, vz
        ])

        self.kalman_filter = KalmanFilter3D(initial_state)
        self.history = [self.centroid.copy()]
        self.time_since_update = 0
        self.age = 1
        self.last_timestamp = None
        self.max_predicted_distance = 1  # Maximum allowed movement per second
        self.confidence = 1.0  # Confidence score for the track
        self.last_reliable_centroid = detection.centroid.copy()  # Store last good position

    def predict(self, current_timestamp):
        if self.last_timestamp is None:
            dt = 0.1  # default value for first prediction
        else:
            dt = (current_timestamp - self.last_timestamp).total_seconds()
        
        # Limit prediction if too much time has passed
        dt = min(dt, 1.0)
        
        # Always run Kalman prediction to maintain state
        predicted_centroid = self.kalman_filter.predict(dt)
        
        # If we haven't updated in a while, use the last reliable position
        if self.time_since_update > 3:  # Adjust this threshold as needed
            self.centroid = self.last_reliable_centroid
        else:
            self.centroid = predicted_centroid
            
        self.time_since_update += 1
        self.age += 1
        self.confidence = max(0.1, 1.0 / (1.0 + self.time_since_update * 0.5))

    def update(self, detection: Detection, timestamp):
        # Calculate distance between prediction and detection
        distance = np.linalg.norm(detection.centroid - self.centroid)
        max_allowed_distance = self.max_predicted_distance * (1 + self.time_since_update * 0.2)
        
        # If detection is close enough to prediction, update normally
        if distance <= max_allowed_distance:
            self.centroid = self.kalman_filter.update(detection.centroid)
            self.last_reliable_centroid = self.centroid.copy()  # Update last reliable position
            self.confidence = 1.0
        else:
            # Bad measurement - keep Kalman running but use last reliable position
            self.kalman_filter.update(self.last_reliable_centroid)  # Update with last good position
            self.centroid = self.last_reliable_centroid
        
        self.history.append(self.centroid.copy())
        self.time_since_update = 0
        self.last_timestamp = timestamp

class TrackedObject:
    def __init__(self, detection: Detection, object_id):
        self.id = object_id
        self.class_label = detection.class_label
        self.centroid = detection.centroid
        
        # Initialize IMM filter with current centroid and zero velocity
        initial_state = np.array([
            detection.centroid[0], 0,  # x, vx
            detection.centroid[1], 0,  # y, vy
            detection.centroid[2], 0   # z, vz
        ])
        
        self.filter = IMMFilter3D(initial_state)
        self.history = [self.centroid.copy()]
        self.time_since_update = 0
        self.age = 1
        self.last_timestamp = None
        self.max_predicted_distance = 0.5
        self.confidence = 1.0
        self.last_reliable_centroid = detection.centroid.copy()

    def predict(self, current_timestamp):
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = (current_timestamp - self.last_timestamp).total_seconds()
        
        dt = min(dt, 1.0)  # Limit maximum time step
        
        predicted_centroid = self.filter.predict(dt)
        
        if self.time_since_update > 3:  # Fallback threshold
            self.centroid = self.last_reliable_centroid
        else:
            self.centroid = predicted_centroid
            
        self.time_since_update += 1
        self.age += 1
        self.confidence = max(0.1, 1.0 / (1.0 + self.time_since_update * 0.5))

    def update(self, detection: Detection, timestamp):
        distance = np.linalg.norm(detection.centroid - self.centroid)
        max_allowed_distance = self.max_predicted_distance * (1 + self.time_since_update * 0.2)
        
        if distance <= max_allowed_distance:
            self.centroid = self.filter.update(detection.centroid)
            self.last_reliable_centroid = self.centroid.copy()
            self.confidence = 1.0
        else:
            self.filter.update(self.last_reliable_centroid)
            self.centroid = self.last_reliable_centroid
        
        self.history.append(self.centroid.copy())
        self.time_since_update = 0
        self.last_timestamp = timestamp

def assign_centroids(previous_detections, current_detections, cost_threshold=10.0, class_mismatch_penalty=1000):
    """
    Assignment function with fallback logic for uncertain matches.
    """
    N = len(previous_detections)
    M = len(current_detections)
    INF_COST = 1e6
    
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))
    
    # Initialize cost matrix
    cost_matrix = np.full((N, M), INF_COST)
    
    # Populate cost matrix with actual costs
    for i, prev_obj in enumerate(previous_detections):
        prev_centroid = prev_obj.centroid
        prev_class = prev_obj.class_label
        
        # Check if we're dealing with a TrackedObject or Detection
        is_tracked = hasattr(prev_obj, 'time_since_update')
        
        for j, curr_detection in enumerate(current_detections):
            curr_centroid = curr_detection.centroid
            curr_class = curr_detection.class_label
            
            # Basic distance cost
            distance = np.linalg.norm(prev_centroid - curr_centroid)
            
            # Different penalties based on class matching
            if prev_class != curr_class:
                cost = INF_COST  # Don't match different classes
            else:
                cost = distance
            
            cost_matrix[i, j] = cost
    
    # Perform the assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_previous = set(range(N))
    unmatched_current = set(range(M))
    
    for i, j in zip(row_ind, col_ind):
        cost = cost_matrix[i, j]
        threshold = cost_threshold
        
        if cost <= threshold and cost < INF_COST:
            matches.append((i, j))
            unmatched_previous.discard(i)
            unmatched_current.discard(j)
    
    return matches, sorted(list(unmatched_previous)), sorted(list(unmatched_current))