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

def assign_centroids(previous_detections, current_detections, cost_threshold=np.inf):
    """
    Assign centroids from the previous frame to the current frame using the Hungarian algorithm.

    Parameters:
    - previous_detections: A list of Detection objects from the previous frame.
    - current_detections: A list of Detection objects from the current frame.
    - cost_threshold: A float representing the maximum allowable cost for assignment.

    Returns:
    - matches: A list of tuples (prev_idx, curr_idx) indicating matched detections.
    - unmatched_previous: A list of indices of detections from the previous frame that were not matched.
    - unmatched_current: A list of indices of detections from the current frame that were not matched.
    """
    N = len(previous_detections)
    M = len(current_detections)

    # Initialize the cost matrix with infinite cost
    cost_matrix = np.full((N, M), np.inf)

    for i, prev_detection in enumerate(previous_detections):
        prev_centroid = prev_detection.centroid
        prev_class = prev_detection.class_label
        for j, curr_detection in enumerate(current_detections):
            curr_centroid = curr_detection.centroid
            curr_class = curr_detection.class_label
            # Only consider matching detections of the same class
            if prev_class == curr_class:
                # Compute Euclidean distance between centroids
                cost = np.linalg.norm(prev_centroid - curr_centroid)
                cost_matrix[i, j] = cost

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_previous = list(range(N))
    unmatched_current = list(range(M))

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] > cost_threshold:
            continue
        matches.append((i, j))
        unmatched_previous.remove(i)
        unmatched_current.remove(j)

    return matches, unmatched_previous, unmatched_current