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

        # self.kalman_filter = KalmanFilter3D_with_cov_est(initial_state,self.class_label,cov_estimator )
        self.kalman_filter = KalmanFilter3D(initial_state)

        self.history = [self.centroid.copy()]
        self.time_since_update = 0
        self.age = 1
        self.last_timestamp = None  # Store the timestamp of last update

    def predict(self, current_timestamp):
        if self.last_timestamp is None:
            dt = 0.1  # default value for first prediction
        else:
            dt = (current_timestamp - self.last_timestamp).total_seconds()
            
        self.centroid = self.kalman_filter.predict(dt)
        self.time_since_update += 1
        self.age += 1
        self.history.append(self.centroid.copy())

    def update(self, detection: Detection, timestamp):
        self.centroid = self.kalman_filter.update(detection.centroid)
        self.history.append(self.centroid.copy())
        self.time_since_update = 0
        self.last_timestamp = timestamp

def assign_centroids(previous_detections, current_detections, cost_threshold=np.inf, class_mismatch_penalty=1000):
    """
    Assigns centroids from previous detections to current detections using the Hungarian algorithm.
    
    Parameters:
    - previous_detections: List of Detection objects from the previous frame.
    - current_detections: List of Detection objects from the current frame.
    - cost_threshold: Maximum allowable cost for a valid match.
    - class_mismatch_penalty: Penalty to add to the cost when class labels don't match.
    
    Returns:
    - matches: List of tuples (prev_idx, curr_idx) of matched detections.
    - unmatched_previous: List of indices of unmatched previous detections.
    - unmatched_current: List of indices of unmatched current detections.
    """
    N = len(previous_detections)
    M = len(current_detections)
    INF_COST = 1e6  # Define a large cost for dummy assignments
    
    if N == 0 and M == 0:
        return [], [], []
    if N == 0:
        matches = []
        unmatched_previous = []
        unmatched_current = list(range(M))
        return matches, unmatched_previous, unmatched_current
    if M == 0:
        matches = []
        unmatched_previous = list(range(N))
        unmatched_current = []
        return matches, unmatched_previous, unmatched_current
    
    # Determine the size of the square cost matrix
    size = max(N, M)
    
    # Initialize the cost matrix with INF_COST
    cost_matrix = np.full((size, size), INF_COST)
    
    # Populate the cost matrix with actual costs
    for i, prev_detection in enumerate(previous_detections):
        prev_centroid = prev_detection.centroid
        prev_class = prev_detection.class_label
        for j, curr_detection in enumerate(current_detections):
            curr_centroid = curr_detection.centroid
            curr_class = curr_detection.class_label
            # Compute Euclidean distance between centroids
            distance = np.linalg.norm(prev_centroid - curr_centroid)
            print(f"Distance between {curr_class} and {prev_class}: ",distance)
            # Add penalty if class labels don't match
            if prev_class != curr_class:
                cost = distance + class_mismatch_penalty
            else:
                cost = distance
            cost_matrix[i, j] = cost
    
    # Perform the assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_previous = set(range(N))
    unmatched_current = set(range(M))
    
    for i, j in zip(row_ind, col_ind):
        if i < N and j < M:
            cost = cost_matrix[i, j]
            if cost <= cost_threshold:
                matches.append((i, j))
                unmatched_previous.discard(i)
                unmatched_current.discard(j)
            # Else, it's an assignment to dummy (INF_COST), treat as unmatched
        # Assignments beyond N or M are dummy assignments, already considered unmatched
    
    # Convert sets to sorted lists
    unmatched_previous = sorted(list(unmatched_previous))
    unmatched_current = sorted(list(unmatched_current))
    
    return matches, unmatched_previous, unmatched_current