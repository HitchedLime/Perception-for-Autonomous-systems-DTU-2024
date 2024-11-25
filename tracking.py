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
        self.age = 1  # Number of frames the object has been tracked
        self.time_since_update = 0
        self.history = [self.centroid]
        # Initialize Kalman filter if you plan to use it later
        # self.kalman_filter = KalmanFilter()

    def predict(self):
        # Predict the next centroid (if using Kalman filter)
        # For now, we can skip this step
        self.time_since_update += 1
        self.age += 1

    def update(self, detection: Detection):
        self.centroid = detection.centroid
        self.history.append(self.centroid)
        self.time_since_update = 0
        # Update Kalman filter here if using


def associate_detections_to_trackers(detections, trackers, threshold = np.inf):
    cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)

    for t, tracker in enumerate(trackers):
        for d, detection in enumerate(detections):
            # Compute the cost (e.g., Euclidean distance)
            cost_matrix[t, d] = np.linalg.norm(tracker.centroid - detection.centroid)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_trackers = list(range(len(trackers)))
    unmatched_detections = list(range(len(detections)))

    for t, d in zip(row_ind, col_ind):
        if cost_matrix[t, d] > threshold:  # Define a cost threshold
            continue
        matches.append((t, d))
        unmatched_trackers.remove(t)
        unmatched_detections.remove(d)

    return matches, unmatched_detections, unmatched_trackers

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