import numpy as np
from scipy.optimize import linear_sum_assignment
from kalmanfilter import KalmanFilter

class Detection:
    def __init__(self, centroid, class_label):
        self.centroid = centroid  # A NumPy array of shape (3,)
        self.class_label = class_label  # An integer or string representing the class

class TrackedObject:
    def __init__(self, detection: Detection, object_id):
        self.id = object_id
        self.class_label = detection.class_label
        self.centroid = detection.centroid
        self.history = [self.centroid.copy()]
        self.time_since_update = 0
        self.age = 1
        # Initialize Kalman filter if used
        # self.kalman_filter = KalmanFilter()

    def predict(self):
        # If using a Kalman filter, predict the next state
        # self.kalman_filter.predict()
        # self.centroid = self.kalman_filter.get_predicted_state()
        self.time_since_update += 1
        self.age += 1
        # For history, you might want to store predictions as well
        # self.history.append(self.centroid.copy())

    def update(self, detection: Detection):
        self.centroid = detection.centroid
        self.history.append(self.centroid.copy())
        self.time_since_update = 0
        # Update Kalman filter with new detection if used
        # self.kalman_filter.update(detection.centroid)

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
    
    if N == 0 or M == 0:
        # No detections to match
        matches = []
        unmatched_previous = list(range(N))
        unmatched_current = list(range(M))
        return matches, unmatched_previous, unmatched_current
    
    # Initialize the cost matrix
    cost_matrix = np.zeros((N, M))
    
    for i, prev_detection in enumerate(previous_detections):
        prev_centroid = prev_detection.centroid
        prev_class = prev_detection.class_label
        for j, curr_detection in enumerate(current_detections):
            curr_centroid = curr_detection.centroid
            curr_class = curr_detection.class_label
            # Compute Euclidean distance between centroids
            distance = np.linalg.norm(prev_centroid - curr_centroid)
            # Add penalty if class labels don't match
            if prev_class != curr_class:
                cost = distance + class_mismatch_penalty
            else:
                cost = distance
            cost_matrix[i, j] = cost
    
    # Apply cost threshold to filter out unlikely matches
    cost_matrix[cost_matrix > cost_threshold] = np.inf
    
    # Check if the cost matrix is feasible
    if np.all(np.isinf(cost_matrix)):
        print("Cost matrix is infeasible (all entries are infinite).")
        matches = []
        unmatched_previous = list(range(N))
        unmatched_current = list(range(M))
        return matches, unmatched_previous, unmatched_current
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_previous = list(range(N))
    unmatched_current = list(range(M))
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == np.inf:
            continue  # Skip assignments with infinite cost
        matches.append((i, j))
        unmatched_previous.remove(i)
        unmatched_current.remove(j)
    
    return matches, unmatched_previous, unmatched_current
